"""MAGNET-style neural tokenizer baseline (heavily scaled down).

Reimplements Ahia et al., "MAGNET: Improving the Multilingual Fairness of
Language Models" (arxiv.org/abs/2407.08818), at much smaller scale. The
downsample/upsample segment-pooling step (common.training.dynamic_pooling) is
directly reused (with the project owner's authorization) from the reference
repo's src/shortening.py (github.com/orevaahia/magnet-tokenization);
flexitokens/model.py shares the same file, since both papers use the same
"dynamic pooling" idea. Everything else (BoundaryPredictor, TransformerBlock,
per-script wiring) is an independent reimplementation, not a port.

Architecture (one forward call handles ONE script's sub-batch -- see
MagnetTrainer in train.py, which splits a mixed-script batch by script first):

  1. byte embedding + learned absolute position embedding.
  2. `pre_layers`: causal self-attention transformer blocks over the full byte
     sequence, so every boundary decision and segment representation is a
     function only of bytes up to that position.
  3. `BoundaryPredictor` (one per SCRIPT, not per language -- languages
     sharing a script, e.g. arz_Arab/kas_Arab, share one predictor and target
     rate; see common.data.oldi_data.LANG_SCRIPT): MLP -> boundary logit ->
     sigmoid -> Gumbel-sigmoid (RelaxedBernoulli) reparameterized soft sample
     -> straight-through hard 0/1 in the forward pass, with the soft value's
     gradient kept for the backward pass (Bengio et al. 2013):
         hard = (soft > 0.5).float()
         boundary = hard - soft.detach() + soft
  4. Downsample (common.training.dynamic_pooling.downsample): mean-pools
     hidden states per predicted segment via a differentiable (B, T, S)
     assignment matrix built from the straight-through boundary tensor --
     gradient flows through the pooling weights themselves, not just the
     pooled values.
  5. `shortened_layers`: more causal transformer blocks over the much shorter
     per-segment sequence.
  6. Upsample (common.training.dynamic_pooling.upsample): broadcasts each
     pooled segment's representation back to its member byte positions
     (transposed assignment matrix), one-segment-shifted for causal safety
     (see below), residual-added to the pre_layers output, then `post_layers`
     and a linear head to a 256-way byte softmax.

Causal safety of the downsample/upsample round-trip: naively broadcasting a
segment's pooled representation to ALL its own byte positions would leak
information from the segment's END into predicting the byte after its
EARLIER positions (e.g. segment [3,4,5]: position 3 would "see" byte 5 when
predicting byte 4). Fix, built into dynamic_pooling itself: a one-segment
SHIFT -- a position receives the pooled representation of the most recently
*closed* segment strictly before it, never its own. The learned
`null_segment` parameter fills in before any segment has closed.

Loss (computed in train.py): next-byte cross-entropy + a per-script
boundary-rate term -- negative log-likelihood of the observed per-sequence
boundary count under Binomial(real_length, prior), `prior` being a
configurable per-script target rate (see MagnetConfig.default_boundary_prior).

Deliberate simplifications vs. the paper (sized to match fairtok's own
BytePolicy compute budget, not a reproduction of the paper's 100M+ parameter
model trained on 10B+ bytes):
  - Scale: d_model in the tens (default 64), 1-2 layers per stage, few
    attention heads (see MagnetConfig in train.py, same order of magnitude as
    fairtok.policy.BytePolicy's hidden_size/num_layers).
  - Positional encoding: plain learned absolute embeddings, not the
    reference's Transformer-XL-style relative attention -- not worth the
    complexity at this scale.
  - Boundary priors: a flat, hand-set per-script hyperparameter (default 0.3,
    ~3.3 bytes/token) rather than measured per-script off a plain-BPE anchor
    (contrast fairtok.train._plain_bpe_target_rate) -- an allowed
    simplification.
  - No temperature annealing: fixed Gumbel-sigmoid temperature
    (MagnetConfig.boundary_temperature) rather than annealed over training.
  - Attention: plain `nn.MultiheadAttention` with causal + padding masks, not
    a hand-rolled relative-attention kernel.
"""

import torch
import torch.nn as nn

from common.training.dynamic_pooling import downsample, upsample


class TransformerBlock(nn.Module):
    """Pre-norm causal self-attention + FFN block, shared unmodified by all
    three stages (pre/shortened/post_layers). The shortened stage must stay
    causal too, or a segment's representation could leak info from later
    segments into upsample's broadcast (see module docstring)."""

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
        """x: (B, L, D). key_padding_mask: (B, L) bool, True = PAD (matches
        nn.MultiheadAttention's own convention)."""
        L = x.size(1)
        # Rebuilt each call since L varies (full byte length vs. shortened
        # segment count); cost is negligible next to the attention matmul.
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

        sample=True (training): RelaxedBernoulli(temperature, probs).rsample()
        -- the Concrete-distribution relaxation (Jang/Maddison 2017),
        differentiable w.r.t. `probs` unlike a plain torch.bernoulli draw.

        sample=False (deterministic inference): skips sampling, uses the raw
        probability as the "soft" value -- matches fairtok.policy.segment_bytes's
        `deterministic` flag (reproducible segmentation, not exploration).
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
        # Straight-through estimator (Bengio et al. 2013): forward value is
        # `hard` (genuine 0/1), but d(boundary)/d(soft) == 1, so gradient
        # flows as if this were just `soft`.
        boundary = hard - soft.detach() + soft
        return logits, probs, boundary


def _seg_valid_mask(hard_boundaries, valid, num_pooled_slots):
    """Which of downsample()'s (B, S+1) pooled slots are real segments per
    batch item vs. padding slots needed only because another sequence in the
    batch had more segments. Slot 0 (null segment) is always real; segments
    fill in order 0..count-1 with no gaps, so "slot index < count" means
    "real" for that item."""
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
    """See module docstring for the architecture. One forward call handles a
    single script's sub-batch; pre/shortened/post stages are shared nn.Module
    instances across scripts, only `boundary_predictors[script]` differs."""

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
        # position embedding, not the reference's relative attention (see
        # module docstring's "positional encoding" simplification)

        self.pre_layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_pre_layers)
            ]
        )

        # One predictor per SCRIPT (see module docstring point 3). ModuleDict
        # keys are fixed at construction; scripts must be known up front since
        # adding a key later would need a fresh optimizer to see the new params.
        self.boundary_predictors = nn.ModuleDict(
            {
                script: BoundaryPredictor(d_model, temperature=boundary_temperature)
                for script in scripts
            }
        )

        self.null_segment = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.down_ln = nn.LayerNorm(d_model)  # normalizes pooled segment reps
        # before the shortened stage -- pooling changes activation scale (mean
        # of a variable number of vectors); matches the reference's placement.
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
        """byte_ids: (B, T) long, zero-padded. lengths: (B,) real length per
        sequence. script: key of self.boundary_predictors governing this whole
        call (caller must split mixed-script batches first). sample: see
        BoundaryPredictor.forward.

        Returns (logits, boundary_probs, hard_boundaries, key_padding_mask):
          logits: (B, T, vocab_size), not yet shifted (caller shifts for
            next-byte prediction).
          boundary_probs: pre-hardening probabilities, diagnostics only.
          hard_boundaries: straight-through 0/1 decisions, used by the
            boundary-rate loss and for vocabulary harvesting.
          key_padding_mask: (B, T) bool, True = PAD.
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        pos_idx = torch.arange(T, device=device)
        key_padding_mask = pos_idx[None, :] >= lengths[:, None]  # True = PAD

        # Clamp is applied to a copy, not pos_idx itself: pos_idx is also used
        # above for key_padding_mask, which needs the true (unclamped)
        # position. Without this clamp, T > max_len (1024 default) makes
        # pos_embed's gather go out of bounds -- silently wrong on CPU, a
        # device-side assert on GPU (reported async at the next sync point,
        # misleadingly pointing elsewhere). Same guard in flexitokens/model.py's
        # TransformerStage.forward.
        clamped_pos_idx = pos_idx.clamp(max=self.max_len - 1)
        x = self.byte_embed(byte_ids) + self.pos_embed(clamped_pos_idx)[None, :, :]
        for layer in self.pre_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        boundary_logits, boundary_probs, hard_boundaries = self.boundary_predictors[
            script
        ](x, sample=sample)
        # Padded positions must never register as boundaries (would corrupt
        # segment ids and inflate the boundary-rate loss). Multiplying by the
        # valid mask AFTER the straight-through construction keeps gradient
        # flow to boundary_logits for real positions while zeroing the
        # forward value in pad.
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
