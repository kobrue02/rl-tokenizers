"""MAGNET-style neural tokenizer baseline (heavily scaled down).

Ahia et al., "MAGNET: Improving the Multilingual Fairness of Language Models"
(arxiv.org/abs/2407.08818). The reference implementation this project studied
(and cloned into a scratch dir for reading, MIT licensed) lives at
github.com/orevaahia/magnet-tokenization -- specifically src/magnet.py
(MagnetTransformerLM, BoundaryPredictor) and src/shortening.py (the
downsample/upsample einsum trick). This module is an independent, much smaller
reimplementation of the same *mechanism*, not a port of that code.

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
     boundary rate; see fairtok.oldi_data.LANG_SCRIPT, which is where the
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
  4. Downsample: mean-pool hidden states within each predicted segment (a run of
     bytes between consecutive boundaries), computed as a batched, vectorized op
     via `torch.cumsum` over the boundary tensor to get a per-position segment id,
     then `scatter_add_`/divide-by-count to get the per-segment mean -- no Python
     loop over positions or segments.
  5. `shortened_layers`: a few more CAUSAL transformer blocks, now over the much
     shorter per-segment sequence.
  6. Upsample: broadcast each pooled segment's (post-shortened-layers)
     representation back out to its member byte positions, via a `gather` on a
     SHIFTED segment id (see "causal safety" below) + a residual add of the
     pre_layers output, then `post_layers` (a couple more causal transformer
     blocks) and a linear head to a 256-way byte softmax.

Causal safety of the downsample/upsample round-trip (this is the one part of
the mechanism that is easy to get subtly wrong, so it's worth spelling out):
naively broadcasting a segment's pooled representation back to ALL of that
segment's own byte positions would leak information from the END of the
segment into predicting the byte immediately after its EARLIER positions (e.g.
if segment = bytes [3,4,5] with a boundary at 5, giving position 3 that
segment's pooled representation lets it "see" byte 5 when predicting byte 4 --
a genuine leak, since byte 5 comes after byte 4). The fix, taken directly from
the cloned reference implementation's shortening.py (see the comment there:
"i-th group can be upsampled only to the tokens from (i+1)-th group, otherwise
there's a leak"), is a one-segment SHIFT: a position receives the pooled
representation of the most recently *closed* segment strictly before it, never
its own (possibly still-open, possibly just-closed) segment. A learned
`null_segment` parameter fills in for positions before any segment has closed
yet (there's nothing causally-safe to give them otherwise). Concretely this
falls out of computing two slightly different cumulative sums of the same
boundary tensor -- see `_segment_ids` below -- rather than needing separate
bookkeeping.

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
  - Segment routing is NOT fully differentiable end-to-end the way the
    reference implementation's dense einsum assignment matrix is (there,
    gradient flows both through which hidden vectors get pooled/broadcast AND
    through the pooling/broadcast WEIGHTS themselves, since the assignment
    matrix is built from arithmetic on the straight-through boundary tensor).
    Here, segment ids are cast to integers for `scatter_add_`/`gather` indexing,
    which severs the gradient path through the routing decision itself --
    only the POOLED VALUES (and, separately and directly, the boundary-rate
    loss acting on the raw straight-through boundary tensor) carry gradient
    back to the boundary predictor. This is a real reduction in signal
    richness, but the boundary-rate loss is precisely the mechanism the task
    specifies for shaping the predictor, so boundary placement is still
    trained end-to-end -- just through one channel instead of two.
  - Attention implementation: plain `nn.MultiheadAttention` with a causal mask
    plus a padding mask, not a hand-rolled relative-attention kernel.
"""

import torch
import torch.nn as nn


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
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class BoundaryPredictor(nn.Module):
    """Per-script MLP -> boundary logit -> Gumbel-sigmoid soft sample ->
    straight-through hard boundary. See module docstring point 3."""

    def __init__(self, d_model, d_hidden=None, temperature=0.5, threshold=0.5):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.net = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
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
            soft = torch.distributions.RelaxedBernoulli(temperature=self.temperature, probs=probs).rsample()
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


def _segment_ids(boundary):
    """boundary: (B, T) straight-through 0/1 float tensor (1 = this position is
    the LAST byte of its segment). Returns (down_id, up_id), each (B, T) long:

    down_id[b, t] = which segment position t belongs to, FOR POOLING (0-indexed,
    counting only segments already started). Subtracting `boundary` itself before
    the cast means the position that CLOSES a segment is still counted as a
    member of the segment it's closing, not bumped into the next one.

    up_id[b, t] = which pooled segment (in the null-prepended pooled sequence,
    see downsample_mean) gets BROADCAST to position t, for UPSAMPLING. NOT
    subtracting `boundary` this time is exactly the one-segment shift that keeps
    the whole thing causally safe (see module docstring): up_id lags down_id by
    one step at every already-closed boundary, which is what makes position t
    receive the most recently CLOSED segment strictly before it, rather than its
    own (possibly not-yet-closed) segment.

    Casting to `.long()` for indexing is where gradient stops flowing through
    the routing decision itself (integer tensors can't require grad) -- see the
    "segment routing is NOT fully differentiable" simplification in the module
    docstring. The raw `boundary` tensor (still carrying the straight-through
    gradient) is used separately and directly by the boundary-rate loss in
    train.py, which is the channel that actually trains the boundary predictor.
    """
    cum = torch.cumsum(boundary, dim=1)
    down_id = (cum - boundary).long()
    up_id = cum.long()
    return down_id, up_id


def downsample_mean(hidden, boundary, null_segment):
    """Mean-pool `hidden` (B, T, D) within each segment induced by `boundary`
    (B, T), vectorized via scatter-add + divide-by-count (no Python loop over
    positions or segments). Returns (pooled_with_null, seg_valid):

      pooled_with_null: (B, S+1, D) -- index 0 is `null_segment` (a learned
      parameter, broadcast to every batch item), indices 1..S are the S real
      segments' mean-pooled hidden states, where S = the largest number of
      segments any sequence in this batch actually has. Shorter sequences (fewer
      segments) get zero-padding in the extra slots -- `seg_valid` marks which
      slots are real, for the shortened_layers' padding mask.

      seg_valid: (B, S+1) bool, True for real (non-padding) segment slots. Slot
      0 (the null segment) is always valid.
    """
    B, T, D = hidden.shape
    down_id, _ = _segment_ids(boundary)
    n_segments = int(down_id.max().item()) + 1 if T > 0 else 0

    pooled = hidden.new_zeros(B, n_segments, D)
    counts = hidden.new_zeros(B, n_segments)
    idx = down_id.unsqueeze(-1).expand(-1, -1, D)
    pooled.scatter_add_(1, idx, hidden)
    counts.scatter_add_(1, down_id, hidden.new_ones(B, T))
    pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)

    null = null_segment.expand(B, 1, D)
    pooled_with_null = torch.cat([null, pooled], dim=1)  # (B, S+1, D)
    seg_valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=hidden.device), counts > 0], dim=1)
    return pooled_with_null, seg_valid


def upsample_broadcast(pooled_with_null, boundary):
    """Broadcast each pooled segment back out to its member byte positions, via
    a `gather` on the SHIFTED segment id (see `_segment_ids`'s up_id -- this is
    what keeps this causally safe rather than a plain per-segment repeat).
    pooled_with_null: (B, S+1, D). boundary: (B, T). Returns (B, T, D)."""
    _, up_id = _segment_ids(boundary)
    up_id = up_id.clamp(max=pooled_with_null.size(1) - 1)  # defensive: up_id's own
    # max is bounded by the number of boundaries actually fired in THIS sequence,
    # which is <= n_segments computed from down_id, so this should never actually
    # clamp anything in practice -- kept as a cheap guard against off-by-one drift
    # rather than a load-bearing correctness fix.
    D = pooled_with_null.size(-1)
    idx = up_id.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(pooled_with_null, 1, idx)


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
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_pre_layers)]
        )

        # One predictor per SCRIPT (see module docstring point 3) -- ModuleDict
        # keys are fixed at construction time (scripts must be known up front,
        # since adding a key later would need a fresh optimizer to see the new
        # parameters).
        self.boundary_predictors = nn.ModuleDict(
            {script: BoundaryPredictor(d_model, temperature=boundary_temperature) for script in scripts}
        )

        self.null_segment = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.down_ln = nn.LayerNorm(d_model)  # normalize pooled segment reps before
        # the shortened stage -- pooling changes the activation scale (mean of a
        # variable number of vectors), LayerNorm re-stabilizes it; the reference
        # implementation does the same (`self.down_ln` in magnet.py) at the same spot.
        self.shortened_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_shortened_layers)]
        )
        self.post_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_post_layers)]
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

        x = self.byte_embed(byte_ids) + self.pos_embed(pos_idx)[None, :, :]
        for layer in self.pre_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        boundary_logits, boundary_probs, hard_boundaries = self.boundary_predictors[script](x, sample=sample)
        # Padded positions must never register as boundaries -- a phantom cut
        # inside the padding region would corrupt the segment ids computed below
        # (scatter/gather indices), and would double-count in the boundary-rate
        # loss (which sums hard_boundaries per sequence). Multiplying by the
        # valid mask AFTER the straight-through construction keeps the gradient
        # path to boundary_logits intact for the real positions while making the
        # forward value exactly 0 in the pad region.
        valid = (~key_padding_mask).float()
        hard_boundaries = hard_boundaries * valid

        pooled_with_null, seg_valid = downsample_mean(x, hard_boundaries, self.null_segment)
        pooled_with_null = self.down_ln(pooled_with_null)
        seg_padding_mask = ~seg_valid
        for layer in self.shortened_layers:
            pooled_with_null = layer(pooled_with_null, key_padding_mask=seg_padding_mask)

        upsampled = upsample_broadcast(pooled_with_null, hard_boundaries)
        h = upsampled + x  # residual connection -- see module docstring step 6
        for layer in self.post_layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        logits = self.output_head(h)
        return logits, boundary_probs, hard_boundaries, key_padding_mask
