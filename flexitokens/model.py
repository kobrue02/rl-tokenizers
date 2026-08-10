"""FlexiTokens-style differentiable byte-level tokenizer.

A from-scratch, CLEAN-ROOM reimplementation of the architecture described in the
FlexiTokens paper (ACL Findings 2026, https://aclanthology.org/2026.findings-acl.848.pdf).
No reference implementation was consulted: the official repo
(github.com/skai-research/flexitokens) ships with no LICENSE file, so it defaults to
all-rights-reserved copyright and was deliberately never cloned, opened, or read while
writing this. Everything below is derived from the paper's own prose/equations only.
Every place where the paper's abstract-level description under-specifies a concrete
mechanism, this file makes an explicit, documented choice -- search for "JUDGMENT CALL".

-------------------------------------------------------------------------------------
SCALE-DOWN NOTICE: the paper trains 126M-388M parameter models on ~56B bytes of data.
This is a baseline sized to compare against THIS project's own compute scale (see
fairtok.policy.BytePolicy -- a few hundred thousand parameters, GRU-based, trained on
a handful of GRPO steps over small batches), not an attempt to reproduce the paper's
real numbers. d_model / layer counts below default to 1-2 orders of magnitude smaller
than the paper's smallest (126M) configuration.
-------------------------------------------------------------------------------------

Architecture (U-Net-shaped, byte-in / byte-out):

  1. Byte embedding (+ learned positional embedding) feeds a "pre" stack of CAUSAL
     transformer encoder layers over the full byte sequence.
  2. Boundary predictor: ONE small shared MLP -- no per-language or per-script
     parameters anywhere in this model at all -- maps each position's "pre" hidden
     state to a scalar boundary logit. That script-agnostic uniformity (not the
     U-Net downsample/upsample shape itself, which prior work like MANTA/MAGNET/
     Hourglass also uses) is FlexiTokens' actual point of contrast with prior
     per-script boundary predictors.
  3. Gumbel-sigmoid relaxation (torch.distributions.RelaxedBernoulli) of that logit
     during training gives a reparameterized, differentiable sample; a
     straight-through estimator hardens it to a real 0/1 decision in the forward
     pass while keeping the soft value's gradient for the backward pass:
         hard = (soft > 0.5).float()
         boundary = hard - soft.detach() + soft
     At inference (`deterministic=True`, see flexitokens/segment.py) no noise is
     sampled at all -- `soft` is just sigmoid(logit), thresholded the same way.
  4. Downsample: mean-pool the "pre" hidden states within each predicted segment.
     Segment ids come from an EXCLUSIVE cumulative sum of the (detached, hardened)
     boundary tensor -- JUDGMENT CALL: shifted right by one position first, so a
     boundary AT position t closes the segment CONTAINING t. This matches
     fairtok.policy.spans_from_boundaries's own convention exactly (boundary=1 at
     position t means "this byte ends a token"), so induced spans behave
     identically whether they came from fairtok's GRU policy or this model. See
     `compute_segment_ids` below. Pooling itself is one `scatter_add_` call --
     no Python loop over positions or segments.
  5. A "mid" stack of causal transformer layers on the pooled (shorter) sequence.
  6. Upsample: gather each pooled segment's representation back out to every byte
     position that segment covers (the exact inverse of step 4's scatter), add it
     as a residual to the ORIGINAL "pre" hidden state (so byte-local detail lost by
     mean-pooling isn't gone for good), run a "post" stack of causal transformer
     layers over the full byte length again, and project to a 256-way next-byte
     softmax.

Loss = next-byte cross-entropy (standard, teacher-forced byte-level LM loss) + a
per-language HINGE/RANGE boundary-rate loss (`boundary_hinge_loss` below). This loss
term -- not the U-Net shape -- is FlexiTokens' actual contribution over hard-target
approaches like MAGNET: instead of imposing one fixed target compression rate on
every language, each language gets a BAND [beta_L, alpha_L] with exactly zero
gradient pressure inside it. A language is only penalized for compressing MORE than
alpha_L (over-fragmentation -- wastes compute per unit of content) or LESS than
beta_L (under-fragmentation -- collapses whole sentences into a few huge,
non-generalizing spans); anywhere between beta_L and alpha_L is free to move
however training likes. That's the "flexible" in FlexiTokens.

JUDGMENT CALL -- no key_padding_mask anywhere below, despite batches being padded to
a common length (see `pad_byte_batch`). This isn't an oversight: every stage uses a
CAUSAL mask, and every batch is padded on the RIGHT. For any real (non-pad) position
p, causal attention already restricts its attention to positions <= p, and since
padding never precedes real content within one sequence, every position <= p is
itself real. The pooled ("mid") stage inherits the same property because pooled
segment index is non-decreasing in byte position: a real segment's index is always
less than the index of any fake all-zero pooled slot created purely by padding's own
(spurious, meaningless) boundary decisions, so causal masking alone keeps real
queries from ever attending those fake slots. A key_padding_mask is only needed for
NON-causal (bidirectional) attention, which this model deliberately never uses --
next-byte prediction requires strict left-to-right causality at every stage anyway.

JUDGMENT CALL -- gradient path into the boundary predictor. Segment ids (step 4) are
plain integers used as scatter/gather indices, and integer indices carry no
gradient. That means the reconstruction (next-byte) loss's gradient does NOT flow
back into WHICH positions get grouped together -- only into the CONTENT of the
representations that get pooled (via h_pre) and, downstream, into the boundary
predictor's weights only through whatever path the straight-through boundary tensor
itself participates in directly. In this implementation that direct path is exactly
`boundary_hinge_loss`: it consumes the continuous straight-through `boundaries`
tensor (not the integer segment ids), so it IS the primary training signal shaping
boundary placement. A fully soft/continuous assignment matrix (as opposed to hard
integer segment ids) could let the reconstruction loss shape boundaries too, but
that is a heavier design not specified by the architecture description this file
implements from (which explicitly calls for a cumsum-based integer segment id +
scatter-mean), so it is not attempted here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

BYTE_VOCAB_SIZE = 256


def pad_byte_batch(byte_seqs, device="cpu"):
    """byte_seqs: list of 1-D LongTensors (variable length). Right-pads with byte
    value 0 up to the batch's longest sequence -- same convention as
    fairtok.policy.batched_sample_rollout's own padding, which is what makes the
    "no key_padding_mask needed" argument in this module's docstring hold (padding
    is always at the tail, never interleaved with or preceding real content).
    Returns (padded: (B,T) long, lengths: (B,) long)."""
    lengths = torch.tensor([int(s.shape[0]) for s in byte_seqs], dtype=torch.long, device=device)
    T = int(lengths.max().item()) if len(byte_seqs) else 0
    padded = torch.zeros(len(byte_seqs), T, dtype=torch.long, device=device)
    for i, seq in enumerate(byte_seqs):
        padded[i, : seq.shape[0]] = seq.to(device)
    return padded, lengths


def _causal_mask(T, device):
    # True = "not allowed to attend" (nn.TransformerEncoder's boolean-mask convention).
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class TransformerStage(nn.Module):
    """A small stack of CAUSAL, pre-norm transformer encoder layers. Used three
    times by FlexiTokensModel (pre/mid/post) at three different sequence
    granularities (full byte length, pooled segment length, full byte length
    again) -- factored out once rather than duplicated three times."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=None, dropout=0.0):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: notably more stable for small/shallow
            # transformers trained from scratch with no warmup schedule, which is
            # exactly this smoke-test-scale setup.
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        T = x.shape[1]
        mask = _causal_mask(T, x.device)
        return self.encoder(x, mask=mask, is_causal=True)


class BoundaryPredictor(nn.Module):
    """The ONE shared boundary-decision MLP -- no per-language, per-script, or
    per-position parameters. Every byte position, in every language, in every
    script, is scored by these exact same weights. This uniformity is FlexiTokens'
    actual point of contrast with prior per-script boundary predictors (e.g.
    MANTA/MAGNET-style approaches that condition on script identity)."""

    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(8, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)  # (B, T) raw logits


def compute_segment_ids(boundaries):
    """boundaries: (B,T) 0/1-valued tensor where boundaries[b,t]==1 means position t
    is the LAST byte of its segment -- matches fairtok.policy.spans_from_boundaries's
    convention exactly (see this module's docstring, point 4).

    Segment id for position t = number of boundaries strictly BEFORE t (an exclusive
    prefix sum), computed by shifting boundaries right by one position first, then
    taking an ordinary inclusive cumsum. Position 0 is always segment 0; if
    boundaries[b,0]==1 (position 0 closes its own segment), position 1 starts fresh
    at segment 1.

    Detaches and casts to long: integer indices are inherently non-differentiable,
    so this is deliberately NOT part of the differentiable graph -- see this
    module's docstring for what that implies for where boundary-predictor gradients
    actually come from."""
    b = boundaries.detach().to(torch.long)
    shifted = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], dim=1)
    return torch.cumsum(shifted, dim=1)


def segment_mean_pool(hidden, segment_ids, valid_mask, num_segments):
    """Vectorized scatter-mean: hidden (B,T,D), segment_ids (B,T) int in
    [0, num_segments), valid_mask (B,T) float/bool marking real (non-pad) positions.

    Padded positions are excluded from BOTH the running sum and the count via
    valid_mask, not merely given a harmless index -- otherwise a segment
    containing the last few real bytes of a sequence (if no boundary happened to
    fall exactly on the sequence's last real byte) would have zero-valued padding
    positions silently averaged into its mean, pulling every trailing segment
    quietly toward zero regardless of what real content it actually contains.

    Returns pooled: (B, num_segments, D)."""
    B, T, D = hidden.shape
    device = hidden.device
    vmask = valid_mask.to(hidden.dtype)
    masked_hidden = hidden * vmask.unsqueeze(-1)

    pooled_sum = torch.zeros(B, num_segments, D, device=device, dtype=hidden.dtype)
    counts = torch.zeros(B, num_segments, device=device, dtype=hidden.dtype)
    idx = segment_ids.unsqueeze(-1).expand(-1, -1, D)
    pooled_sum.scatter_add_(1, idx, masked_hidden)
    counts.scatter_add_(1, segment_ids, vmask)
    return pooled_sum / counts.clamp(min=1.0).unsqueeze(-1)


def next_byte_loss(logits, byte_ids, valid_mask):
    """Standard teacher-forced next-byte cross-entropy: logits[:, t] predicts
    byte_ids[:, t+1]. Masked by valid_mask shifted the same way, so padded
    positions (and the final real position's "next byte", which doesn't exist)
    never contribute. Returns (scalar mean loss over real positions, the mask
    used) -- the mask is returned mainly so callers can compute bits-per-byte or
    accuracy against the same denominator without recomputing it."""
    if logits.shape[1] < 2:
        return logits.sum() * 0.0, valid_mask[:, :0]
    pred = logits[:, :-1, :].reshape(-1, BYTE_VOCAB_SIZE)
    target = byte_ids[:, 1:].reshape(-1)
    mask = valid_mask[:, 1:].reshape(-1)
    per_position = F.cross_entropy(pred, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return per_position.sum() / denom, mask


def boundary_hinge_loss(boundaries, valid_mask, langs, alpha_by_lang, beta_by_lang, default_alpha, default_beta):
    """The paper's per-language hinge/range boundary-rate loss:

        L_BP(lang) = max(rate - alpha_L, 0) + max(beta_L - rate, 0)

    where rate = k/N is the OBSERVED fraction of real byte positions this batch
    marked as boundaries for `lang` (k = sum of boundary decisions, N = number of
    real positions), and [beta_L, alpha_L] is that language's target band (see
    flexitokens/train.py's derive_alpha_beta for how alpha_L/beta_L are set).
    Exactly zero inside the band -- no gradient pressure to hit one fixed rate,
    which is the entire "flexible" premise of FlexiTokens as opposed to a
    hard-target approach like MAGNET.

    `boundaries` is the continuous straight-through tensor from
    FlexiTokensModel.forward (hard-valued in the forward pass, soft-gradiented in
    the backward pass) -- NOT the integer segment ids, which carry no gradient at
    all (see compute_segment_ids). This is deliberately the ONLY loss term in this
    module that touches `boundaries` directly, which is what makes it the primary
    training signal for the boundary predictor's weights (see this module's
    top-level docstring, "gradient path into the boundary predictor").

    langs: list of length B (one language string per sequence in the batch, index
    -aligned with boundaries/valid_mask's batch dimension). Rate is pooled across
    every sequence belonging to the same language in this batch (not one hinge
    term per sequence), then the per-language hinge terms are averaged with EQUAL
    weight per language regardless of how many sequences/bytes that language
    contributed to the batch -- so a batch that happens to sample more high
    -resource-language sequences doesn't dominate the boundary loss, matching this
    whole project's general stance on not letting resource imbalance drive the
    training signal.

    Returns (scalar mean hinge loss, {lang: observed_rate} for logging).
    """
    device = boundaries.device
    per_lang_rate = {}
    losses = []
    unique_langs = sorted(set(langs))
    for lang in unique_langs:
        idx = torch.tensor([i for i, l in enumerate(langs) if l == lang], device=device, dtype=torch.long)
        b = boundaries.index_select(0, idx)
        m = valid_mask.index_select(0, idx)
        denom = m.sum().clamp(min=1.0)
        rate = (b * m).sum() / denom
        alpha = alpha_by_lang.get(lang, default_alpha)
        beta = beta_by_lang.get(lang, default_beta)
        loss = torch.clamp(rate - alpha, min=0.0) + torch.clamp(beta - rate, min=0.0)
        losses.append(loss)
        per_lang_rate[lang] = float(rate.detach().item())
    total = torch.stack(losses).mean() if losses else torch.zeros((), device=device)
    return total, per_lang_rate


class FlexiTokensModel(nn.Module):
    """See this module's top-level docstring for the full architecture
    description and every judgment call. Kept intentionally small (defaults:
    d_model=64, 2 layers per stage) -- comparable in scale to
    fairtok.policy.BytePolicy, not the paper's 126M+ parameter configurations."""

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_pre_layers=2,
        num_mid_layers=2,
        num_post_layers=2,
        dim_feedforward=None,
        max_position_embeddings=512,
        gumbel_temperature=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.gumbel_temperature = gumbel_temperature
        self.max_position_embeddings = max_position_embeddings

        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(max_position_embeddings, d_model)

        self.pre = TransformerStage(d_model, nhead, num_pre_layers, dim_feedforward, dropout)
        self.boundary_predictor = BoundaryPredictor(d_model)
        self.mid = TransformerStage(d_model, nhead, num_mid_layers, dim_feedforward, dropout)
        self.post = TransformerStage(d_model, nhead, num_post_layers, dim_feedforward, dropout)
        self.byte_head = nn.Linear(d_model, BYTE_VOCAB_SIZE)

    def forward(self, byte_ids, lengths, deterministic=False):
        """byte_ids: (B,T) long, right-padded with 0 (see pad_byte_batch). lengths:
        (B,) long, real (non-pad) length per sequence. deterministic=False samples
        the Gumbel-sigmoid relaxation (training); deterministic=True uses a plain
        sigmoid threshold with no noise (inference -- see flexitokens/segment.py).

        Returns a dict: logits (B,T,256), boundaries (B,T) straight-through 0/1
        (gradiented), valid_mask (B,T) float, segment_ids (B,T) long,
        num_segments (int)."""
        B, T = byte_ids.shape
        device = byte_ids.device

        positions = torch.arange(T, device=device).clamp(max=self.max_position_embeddings - 1)
        x = self.byte_embed(byte_ids) + self.pos_embed(positions).unsqueeze(0)

        h_pre = self.pre(x)  # (B,T,D)

        boundary_logits = self.boundary_predictor(h_pre)  # (B,T)
        if deterministic:
            soft = torch.sigmoid(boundary_logits)
        else:
            temperature = torch.tensor(self.gumbel_temperature, device=device, dtype=boundary_logits.dtype)
            soft = RelaxedBernoulli(temperature=temperature, logits=boundary_logits).rsample()
        hard = (soft > 0.5).float()
        boundaries = hard - soft.detach() + soft  # straight-through estimator

        valid_mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()

        # Dynamic segment count: only allocate as many pooled slots as the batch's
        # busiest REAL sequence actually needs, not a flat T -- a fixed T would
        # force the "mid" transformer stage to attend over mostly-empty slots for
        # every sequence shorter or coarser than the batch's longest/finest one.
        raw_segment_ids = compute_segment_ids(boundaries)
        masked_ids = torch.where(valid_mask.bool(), raw_segment_ids, torch.zeros_like(raw_segment_ids))
        num_segments = int(masked_ids.max().item()) + 1 if T > 0 else 1
        # Clamp (not just for real positions -- for ALL positions, including
        # padding): padded positions can have segment ids past num_segments-1
        # since padding can carry its own (meaningless) predicted boundaries, and
        # scatter/gather need every index in-bounds regardless of whether the
        # value at that position is ever used. Their contribution is exactly zero
        # either way (valid_mask=0 zeroes them out in segment_mean_pool), so
        # clamping them into range is safe and changes no real segment's value.
        segment_ids = raw_segment_ids.clamp(max=num_segments - 1)

        pooled = segment_mean_pool(h_pre, segment_ids, valid_mask, num_segments)  # (B, S, D)
        mid_out = self.mid(pooled)  # (B, S, D)

        gathered = torch.gather(mid_out, 1, segment_ids.unsqueeze(-1).expand(-1, -1, self.d_model))
        upsampled = gathered + h_pre  # residual: byte-local detail the mean-pool discarded

        post_out = self.post(upsampled)
        logits = self.byte_head(post_out)

        return {
            "logits": logits,
            "boundaries": boundaries,
            "valid_mask": valid_mask,
            "segment_ids": segment_ids,
            "num_segments": num_segments,
        }
