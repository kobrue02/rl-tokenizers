"""MAGNET-style neural tokenizer baseline (heavily scaled down).

Ahia et al., "MAGNET: Improving the Multilingual Fairness of Language Models"
(arxiv.org/abs/2407.08818). The reference implementation (MIT licensed) lives at
github.com/orevaahia/magnet-tokenization -- specifically src/magnet.py
(MagnetTransformerLM, BoundaryPredictor) and src/shortening.py. The overall
architecture below (embed -> pre_layers -> boundary predictor -> downsample ->
shortened_layers -> upsample -> post_layers -> byte head) is an independent,
much smaller reimplementation of that paper's design, scaled to this project's
compute budget -- but the downsample/upsample segment-pooling step itself
(common.dynamic_pooling) is DIRECTLY REUSED from src/shortening.py (reused with
the project owner's explicit authorization; see also flexitokens/model.py,
which shares the exact same file -- both papers build on the same "dynamic
pooling" lineage). Everything else (BoundaryPredictor, TransformerBlock, the
per-script wiring) is this project's own code, not a port.

Architecture (one forward call handles ONE script's sub-batch -- see
MagnetTrainer in train.py for why a mixed-script training batch gets split by
script before calling this):

  1. byte embedding + learned absolute position embedding
  2. `pre_layers`: a few CAUSAL self-attention transformer blocks over the full
     byte sequence -- this is what makes every downstream boundary decision and
     every downstream segment representation causally safe (a function of bytes
     up to and including the position in question, never later ones).
  3. `BoundaryPredictor` (one per SCRIPT, not per language -- languages that
     share a script, e.g. arz_Arab/kas_Arab, share one predictor and one target
     boundary rate; see common.oldi_data.LANG_SCRIPT, which is where the
     lang -> script mapping is read from in train.py): a small MLP maps each
     position's hidden state to a boundary logit, sigmoid to a probability, then
     Gumbel-sigmoid (RelaxedBernoulli) reparameterized sampling gives a *soft*
     boundary value with a real gradient path, and a straight-through estimator
     hardens it to a genuine 0/1 in the forward pass while keeping that gradient
     path intact for the backward pass:
         hard = (soft > 0.5).float()
         boundary = hard - soft.detach() + soft
     (Bengio et al. 2013's straight-through trick; this is the exact formula the
     cloned reference implementation's BoundaryPredictor.forward uses too.)
  4. Downsample (common.dynamic_pooling.downsample, ported directly from the
     reference's shortening.py): mean-pool hidden states within each predicted
     segment via a dense, differentiable (B, T, S) assignment matrix built from
     the straight-through boundary tensor -- gradient flows through the pooling
     WEIGHTS themselves, not just the pooled values, which a hard-integer-index
     scatter/gather approach (this module's own earlier version) cannot do.
  5. `shortened_layers`: a few more CAUSAL transformer blocks, now over the much
     shorter per-segment sequence.
  6. Upsample (common.dynamic_pooling.upsample, same file): broadcast each
     pooled segment's (post-shortened-layers) representation back out to its
     member byte positions via the same kind of differentiable assignment
     matrix (transposed), one-segment-shifted for causal safety (see below) +
     a residual add of the pre_layers output, then `post_layers` (a couple
     more causal transformer blocks) and a linear head to a 256-way byte
     softmax.

Causal safety of the downsample/upsample round-trip (this is the one part of
the mechanism that is easy to get subtly wrong, so it's worth spelling out):
naively broadcasting a segment's pooled representation back to ALL of that
segment's own byte positions would leak information from the END of the
segment into predicting the byte immediately after its EARLIER positions (e.g.
if segment = bytes [3,4,5] with a boundary at 5, giving position 3 that
segment's pooled representation lets it "see" byte 5 when predicting byte 4 --
a genuine leak, since byte 5 comes after byte 4). The fix, built into
common.dynamic_pooling.downsample/upsample directly (see the comment there:
"segment i's pooled representation is broadcast only to segment i+1's byte
positions, never its own"), is a one-segment SHIFT: a position receives the
pooled representation of the most recently *closed* segment strictly before
it, never its own (possibly still-open, possibly just-closed) segment. The
learned `null_segment` parameter (passed as `downsample`'s `null_group` arg)
fills in for positions before any segment has closed yet.

Loss (computed in train.py, not here): next-byte cross-entropy on the
reconstructed byte sequence, plus a per-script boundary-rate term -- the
negative log-likelihood of the observed per-sequence boundary count under
Binomial(real_length, prior), where `prior` is a configurable per-script target
boundary probability (see train.py's MagnetConfig.default_boundary_prior).

Deliberate simplifications vs. the paper / reference implementation (this is a
baseline sized to match fairtok's own BytePolicy compute budget, not a
reproduction of the paper's 100M+ parameter model trained on 10B+ bytes):

  - Scale: d_model in the tens (default 64), 1-2 layers per stage, a handful of
    attention heads -- vs. the paper's much larger model. See MagnetConfig in
    train.py for exact defaults, chosen to be the same order of magnitude as
    fairtok.policy.BytePolicy's own hidden_size/num_layers.
  - Positional encoding: plain learned absolute position embeddings, not the
    reference implementation's Transformer-XL-style relative positional
    attention (RelPartialLearnableMultiHeadAttn in the cloned magnet.py). At
    this scale and these sequence lengths the extra machinery isn't worth the
    complexity; a real production-scale reproduction would want it back.
  - Boundary priors: exposed as a flat, hand-set per-script hyperparameter
    (default 0.3, i.e. ~3.3 bytes/token on average -- roughly the ballpark a
    real BPE tokenizer lands in) rather than measured per-script from a plain-
    BPE anchor the way fairtok.train._plain_bpe_target_rate derives its single
    global target_rate. Doing the equivalent per-script measurement here is
    straightforward future work; the task explicitly allows this simplification.
  - No temperature annealing: the paper/reference anneal the Gumbel-sigmoid
    temperature over training (sharpening the relaxation as training
    progresses); here it's a fixed hyperparameter (MagnetConfig.boundary_temperature).
  - Attention implementation: plain `nn.MultiheadAttention` with a causal mask
    plus a padding mask, not a hand-rolled relative-attention kernel.

(Segment routing WAS a simplification here -- an earlier version cast segment
ids to integers for scatter_add_/gather indexing, which severed the gradient
path through the routing decision itself, leaving the boundary-rate loss as
the only channel training the boundary predictor. Now that downsample/upsample
are directly reused from the reference implementation, gradient flows through
the pooling weights too, matching the reference exactly -- see point 4/6 above.)
"""

import torch
import torch.nn as nn

from common.dynamic_pooling import downsample, upsample


class TransformerBlock(nn.Module):
    """Pre-norm causal self-attention + FFN block. Used, unmodified, for all
    three stages (pre_layers, shortened_layers, post_layers) -- MAGNET's
    reference implementation applies the exact same causal masking function to
    all three stages too (see magnet.py's `_forward`, shared by every call
    site), which is required here for the same reason: the shortened stage's
    self-attention must ALSO stay causal, or a segment's post-shortened-layer
    representation could mix in information from later segments, which would
    then leak into upsample's broadcast (see module docstring)."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        """x: (B, L, D). key_padding_mask: (B, L) bool, True = position is PAD
        (matches nn.MultiheadAttention's own convention, so it's passed straight
        through with no inversion)."""
        L = x.size(1)
        # Rebuilt every call rather than cached: L varies between calls (full byte
        # length for pre/post_layers, much shorter segment count for
        # shortened_layers), and at this model scale the cost of a fresh
        # (L, L) triu is negligible next to the attention matmul itself.
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class BoundaryPredictor(nn.Module):
    """Per-script MLP -> boundary logit -> Gumbel-sigmoid soft sample ->
    straight-through hard boundary. See module docstring point 3."""

    def __init__(self, d_model, d_hidden=None, temperature=0.5, threshold=0.5):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1)
        )
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, hidden, sample=True):
        """hidden: (B, T, D). Returns (logits, probs, boundary), each (B, T).

        sample=True (training): draws from RelaxedBernoulli(temperature, probs)
        -- the reparameterized ("Concrete distribution", Jang et al. 2017 /
        Maddison et al. 2017) relaxation of a Bernoulli draw, whose `.rsample()`
        is differentiable w.r.t. `probs` (an ordinary `torch.bernoulli` draw is
        not -- there'd be nothing for autograd to differentiate through).

        sample=False (deterministic inference, see segment.py): skips sampling
        entirely and uses the raw probability as the "soft" value -- matching
        fairtok.policy.segment_bytes's own `deterministic` flag, for the same
        reason (reproducible segmentation of a frozen model, not exploration).
        """
        logits = self.net(hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        if sample:
            soft = torch.distributions.RelaxedBernoulli(
                temperature=self.temperature, probs=probs
            ).rsample()
        else:
            soft = probs
        hard = (soft > self.threshold).float()
        # Straight-through estimator (Bengio et al. 2013): the FORWARD value is
        # `hard` (a genuine 0/1, so segment membership downstream is unambiguous),
        # but d(boundary)/d(soft) == 1 everywhere, since `hard` and `soft.detach()`
        # are both constants w.r.t. autograd once detached -- so gradient flows
        # backward as if this had just been `soft`.
        boundary = hard - soft.detach() + soft
        return logits, probs, boundary


def _seg_valid_mask(hard_boundaries, valid, num_pooled_slots):
    """Which of downsample()'s (B, S+1) pooled slots are real segments for each
    batch item, vs. padding slots that exist only because ANOTHER sequence in
    this batch needed more segments. Slot 0 (the null segment, see
    common.dynamic_pooling.downsample) is always real. A batch item's own real
    segment count is exactly how many boundaries it fired among its REAL (non-
    padding) positions -- segments are filled in order 0..count-1 with no gaps,
    so "slot index < count" is exactly "slot is real" for that item."""
    real_segment_count = (hard_boundaries * valid).sum(dim=1, keepdim=True)  # (B, 1)
    slot_idx = torch.arange(
        num_pooled_slots - 1, device=hard_boundaries.device
    ).unsqueeze(
        0
    )  # (1, S)
    real_slots = slot_idx < real_segment_count  # (B, S)
    null_slot = torch.ones(
        hard_boundaries.size(0), 1, dtype=torch.bool, device=hard_boundaries.device
    )
    return torch.cat([null_slot, real_slots], dim=1)  # (B, S+1)


class MagnetModel(nn.Module):
    """See module docstring for the full architecture description. One forward
    call handles a single script's sub-batch (see train.py's MagnetTrainer for
    why a mixed-script training batch is split by script before calling this) --
    the pre/shortened/post transformer stages are shared nn.Module instances
    reused for every script's forward call; only `boundary_predictors[script]`
    differs."""

    def __init__(
        self,
        scripts,
        vocab_size=256,
        d_model=64,
        n_heads=4,
        d_ff=None,
        n_pre_layers=2,
        n_shortened_layers=1,
        n_post_layers=1,
        max_len=1024,
        boundary_temperature=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.byte_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  # plain learned absolute
        # position embedding -- see module docstring's "positional encoding"
        # simplification for why this isn't the reference's relative attention

        self.pre_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_pre_layers)
            ]
        )

        # One predictor per SCRIPT (see module docstring point 3) -- ModuleDict
        # keys are fixed at construction time (scripts must be known up front,
        # since adding a key later would need a fresh optimizer to see the new
        # parameters).
        self.boundary_predictors = nn.ModuleDict(
            {
                script: BoundaryPredictor(d_model, temperature=boundary_temperature)
                for script in scripts
            }
        )

        self.null_segment = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.down_ln = nn.LayerNorm(d_model)  # normalize pooled segment reps before
        # the shortened stage -- pooling changes the activation scale (mean of a
        # variable number of vectors), LayerNorm re-stabilizes it; the reference
        # implementation does the same (`self.down_ln` in magnet.py) at the same spot.
        self.shortened_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_shortened_layers)
            ]
        )
        self.post_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_post_layers)
            ]
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, byte_ids, lengths, script, sample=True):
        """byte_ids: (B, T) long, zero-padded. lengths: (B,) long, real
        (unpadded) length of each sequence. script: str, a key of
        self.boundary_predictors, selects which predictor governs this WHOLE
        call (see class docstring -- a mixed-script batch must be split by
        script by the caller before this is invoked). sample: see
        BoundaryPredictor.forward.

        Returns (logits, boundary_probs, hard_boundaries, key_padding_mask):
          logits: (B, T, vocab_size) -- byte prediction logits, NOT yet shifted
            (train.py's caller does the next-byte shift itself).
          boundary_probs: (B, T) -- pre-hardening sigmoid probabilities, for
            logging/diagnostics only (not used in any loss).
          hard_boundaries: (B, T) -- straight-through 0/1 boundary decisions,
            used directly by train.py's boundary-rate loss and by
            fairtok.policy.spans_from_boundaries for vocabulary harvesting.
          key_padding_mask: (B, T) bool, True = PAD.
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        pos_idx = torch.arange(T, device=device)
        key_padding_mask = pos_idx[None, :] >= lengths[:, None]  # True = PAD

        # clamp BEFORE indexing pos_embed, not after -- pos_idx is also used above
        # for key_padding_mask, which needs the true (unclamped) position to compare
        # against `lengths` correctly. Without this clamp, any sequence longer than
        # self.max_len (1024 by default) makes T > max_len, and pos_embed(pos_idx) --
        # an nn.Embedding(max_len, d_model) -- does an out-of-bounds gather on CUDA
        # (a silent, uncaught index error on CPU, but a hard "vectorized_gather_kernel
        # index out of bounds" device-side assert on GPU, reported asynchronously
        # at the next synchronizing call, which made it look like the CRASH SITE was
        # somewhere unrelated further down the model). flexitokens/model.py's
        # TransformerStage.forward already guards this exact case the same way.
        clamped_pos_idx = pos_idx.clamp(max=self.max_len - 1)
        x = self.byte_embed(byte_ids) + self.pos_embed(clamped_pos_idx)[None, :, :]
        for layer in self.pre_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        boundary_logits, boundary_probs, hard_boundaries = self.boundary_predictors[
            script
        ](x, sample=sample)
        # Padded positions must never register as boundaries -- a phantom cut
        # inside the padding region would corrupt the segment ids computed below
        # (scatter/gather indices), and would double-count in the boundary-rate
        # loss (which sums hard_boundaries per sequence). Multiplying by the
        # valid mask AFTER the straight-through construction keeps the gradient
        # path to boundary_logits intact for the real positions while making the
        # forward value exactly 0 in the pad region.
        valid = (~key_padding_mask).float()
        hard_boundaries = hard_boundaries * valid

        pooled_with_null = downsample(hard_boundaries, x, self.null_segment)
        pooled_with_null = self.down_ln(pooled_with_null)
        seg_valid = _seg_valid_mask(hard_boundaries, valid, pooled_with_null.size(1))
        seg_padding_mask = ~seg_valid
        for layer in self.shortened_layers:
            pooled_with_null = layer(
                pooled_with_null, key_padding_mask=seg_padding_mask
            )

        upsampled = upsample(hard_boundaries, pooled_with_null)
        h = upsampled + x  # residual connection -- see module docstring step 6
        for layer in self.post_layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        logits = self.output_head(h)
        return logits, boundary_probs, hard_boundaries, key_padding_mask
