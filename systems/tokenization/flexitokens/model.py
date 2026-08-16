"""FlexiTokens-style differentiable byte-level tokenizer.

Architecture from the FlexiTokens paper (ACL Findings 2026,
https://aclanthology.org/2026.findings-acl.848.pdf). Originally a clean-room
reimplementation (the reference repo, github.com/skai-research/flexitokens,
has no LICENSE and was never opened while building this). The project owner
has since authorized reusing that repo's code, so the downsample/upsample
segment-pooling step (common.training.dynamic_pooling) is now directly reused
from its src/model/shortening.py. Everything else (BoundaryPredictor,
TransformerStage, boundary_hinge_loss, per-language band derivation in
train.py) remains this project's own code from the paper's description, not
the reference. Places where the paper under-specifies a mechanism are marked
"JUDGMENT CALL".

SCALE-DOWN: the paper trains 126M-388M parameter models on ~56B bytes. This
baseline is sized to match this project's own compute scale (see
fairtok.policy.BytePolicy: a few hundred thousand parameters, GRU-based), not
to reproduce the paper's numbers -- d_model/layer counts default 1-2 orders
of magnitude below the paper's smallest (126M) configuration.

Architecture (U-Net-shaped, byte-in / byte-out):
  1. Byte + learned positional embedding feeds a "pre" stack of CAUSAL
     transformer encoder layers over the full byte sequence.
  2. Boundary predictor: ONE small shared MLP, no per-language or per-script
     parameters at all, maps each "pre" hidden state to a boundary logit.
     That script-agnostic uniformity (not the U-Net shape itself, which
     MANTA/MAGNET/Hourglass also use) is FlexiTokens' actual contrast with
     prior per-script boundary predictors.
  3. Gumbel-sigmoid (RelaxedBernoulli) relaxation of that logit during
     training gives a differentiable sample; a straight-through estimator
     hardens it to 0/1 in the forward pass while keeping the soft gradient
     for backward:
         hard = (soft > 0.5).float()
         boundary = hard - soft.detach() + soft
     At inference (deterministic=True) no noise is sampled -- `soft` is just
     sigmoid(logit).
  4. Downsample (common.training.dynamic_pooling.downsample): mean-pool "pre"
     hidden states per predicted segment via a differentiable (B,T,S)
     assignment matrix from the straight-through boundary tensor via
     cumulative sum. boundary=1 at position t means "byte t ends a token,"
     matching common.bytes_utils.spans_from_boundaries's convention exactly.
  5. A "mid" stack of causal transformer layers on the pooled sequence.
  6. Upsample (common.training.dynamic_pooling.upsample): broadcast each
     pooled segment's representation to every byte position it covers,
     one-segment-shifted for causal safety, residual-added to the original
     "pre" hidden state, run through a "post" causal transformer stack, and
     project to a 256-way next-byte softmax.

Loss = next-byte cross-entropy + a per-language HINGE/RANGE boundary-rate
loss (`boundary_hinge_loss` below) -- FlexiTokens' actual contribution over
hard-target approaches like MAGNET: each language gets a band [beta_L,
alpha_L] with zero gradient pressure inside it, only penalized for
compressing more than alpha_L (over-fragmentation) or less than beta_L
(collapses sentences into huge, non-generalizing spans). That's the
"flexible" in FlexiTokens.

JUDGMENT CALL -- no key_padding_mask anywhere below, despite right-padded
batches. Every stage uses a CAUSAL mask, so for any real position p,
attention is already restricted to positions <= p, and since padding never
precedes real content, every position <= p is real. The pooled ("mid")
stage inherits this because pooled segment index is non-decreasing in byte
position: a real segment's index is always below any padding-only pooled
slot. key_padding_mask would only matter for non-causal attention, which
this model never uses.

Gradient path into the boundary predictor: since downsample/upsample build
their assignment matrix directly from the continuous straight-through
`boundaries` tensor (never detached to integer ids), the reconstruction
loss's gradient flows through the pooling itself into the boundary
predictor's weights, not just through boundary_hinge_loss's direct use of
`boundaries`. (An earlier version hardened boundaries into integer
scatter/gather indices, severing this channel and leaving the hinge loss as
the only training signal.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from common.training.dynamic_pooling import downsample, upsample

BYTE_VOCAB_SIZE = 256


def pad_byte_batch(byte_seqs, device="cpu"):
    """byte_seqs: list of 1-D LongTensors (variable length). Right-pads with 0
    to the batch's longest sequence -- matches
    fairtok.policy.batched_sample_rollout's convention, which is what makes
    the module docstring's "no key_padding_mask needed" argument hold
    (padding is always at the tail). Returns (padded: (B,T) long,
    lengths: (B,) long)."""
    lengths = torch.tensor(
        [int(s.shape[0]) for s in byte_seqs], dtype=torch.long, device=device
    )
    T = int(lengths.max().item()) if len(byte_seqs) else 0
    padded = torch.zeros(len(byte_seqs), T, dtype=torch.long, device=device)
    for i, seq in enumerate(byte_seqs):
        padded[i, : seq.shape[0]] = seq.to(device)
    return padded, lengths


def _causal_mask(T, device):
    # True = "not allowed to attend" (nn.TransformerEncoder's boolean-mask convention).
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class TransformerStage(nn.Module):
    """Stack of causal, pre-norm transformer encoder layers, used three times
    by FlexiTokensModel (pre/mid/post) at three granularities (full byte
    length, pooled segment length, full byte length again)."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=None, dropout=0.0):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for small/shallow
            # transformers trained from scratch without a long warmup.
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        T = x.shape[1]
        mask = _causal_mask(T, x.device)
        return self.encoder(x, mask=mask, is_causal=True)


class BoundaryPredictor(nn.Module):
    """The ONE shared boundary-decision MLP -- no per-language/per-script/
    per-position parameters; every position, in every language, is scored by
    the same weights. This uniformity is FlexiTokens' actual contrast with
    prior per-script predictors (e.g. MANTA/MAGNET)."""

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


def next_byte_loss(logits, byte_ids, valid_mask):
    """Teacher-forced next-byte CE: logits[:, t] predicts byte_ids[:, t+1].
    Masked so padding and each sequence's final (non-existent next-byte)
    position never contribute. Returns (mean loss over real positions, the
    mask) -- mask is returned so callers can reuse the same denominator for
    bits-per-byte/accuracy."""
    if logits.shape[1] < 2:
        return logits.sum() * 0.0, valid_mask[:, :0]
    pred = logits[:, :-1, :].reshape(-1, BYTE_VOCAB_SIZE)
    target = byte_ids[:, 1:].reshape(-1)
    mask = valid_mask[:, 1:].reshape(-1)
    per_position = F.cross_entropy(pred, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return per_position.sum() / denom, mask


def boundary_hinge_loss(
    boundaries,
    valid_mask,
    langs,
    alpha_by_lang,
    beta_by_lang,
    default_alpha,
    default_beta,
):
    """Per-language hinge/range boundary-rate loss from the paper:

        L_BP(lang) = max(rate - alpha_L, 0) + max(beta_L - rate, 0)

    rate = k/N, the observed boundary fraction over real positions for
    `lang`; [beta_L, alpha_L] is that language's target band (see train.py's
    derive_alpha_beta). Zero inside the band -- no pressure toward one fixed
    rate, the "flexible" premise vs. a hard-target approach like MAGNET.

    `boundaries` is the continuous straight-through tensor (gradiented), not
    the integer segment ids -- this is the only loss term touching it
    directly, so it's the primary training signal for the boundary
    predictor's weights.

    langs: length-B list, one language per batch sequence. Rate is pooled
    per language across the batch, then hinge terms are averaged with EQUAL
    weight per language regardless of how many sequences/bytes it
    contributed -- so a batch with more high-resource-language sequences
    doesn't dominate.

    Returns (scalar mean hinge loss, {lang: observed_rate} for logging).
    """
    device = boundaries.device
    per_lang_rate = {}
    losses = []
    unique_langs = sorted(set(langs))
    for lang in unique_langs:
        idx = torch.tensor(
            [i for i, l in enumerate(langs) if l == lang],
            device=device,
            dtype=torch.long,
        )
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
    """See module docstring for architecture and judgment calls. Kept small
    (default d_model=64, 2 layers/stage) -- comparable to
    fairtok.policy.BytePolicy, not the paper's 126M+ parameter configs."""

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

        self.pre = TransformerStage(
            d_model, nhead, num_pre_layers, dim_feedforward, dropout
        )
        self.boundary_predictor = BoundaryPredictor(d_model)
        self.null_group = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # segment-0
        # placeholder for downsample/upsample -- feeds the first real segment's
        # own byte positions before any segment has closed (causal safety).
        self.mid = TransformerStage(
            d_model, nhead, num_mid_layers, dim_feedforward, dropout
        )
        self.post = TransformerStage(
            d_model, nhead, num_post_layers, dim_feedforward, dropout
        )
        self.byte_head = nn.Linear(d_model, BYTE_VOCAB_SIZE)

    def forward(self, byte_ids, lengths, deterministic=False):
        """byte_ids: (B,T) long, right-padded with 0. lengths: (B,) real
        length per sequence. deterministic=False samples the Gumbel-sigmoid
        relaxation (training); deterministic=True thresholds a plain sigmoid,
        no noise (inference).

        Returns dict: logits (B,T,256), boundaries (B,T) straight-through 0/1
        (gradiented), valid_mask (B,T) float."""
        B, T = byte_ids.shape
        device = byte_ids.device

        positions = torch.arange(T, device=device).clamp(
            max=self.max_position_embeddings - 1
        )
        x = self.byte_embed(byte_ids) + self.pos_embed(positions).unsqueeze(0)

        h_pre = self.pre(x)  # (B,T,D)

        boundary_logits = self.boundary_predictor(h_pre)  # (B,T)
        if deterministic:
            soft = torch.sigmoid(boundary_logits)
        else:
            temperature = torch.tensor(
                self.gumbel_temperature, device=device, dtype=boundary_logits.dtype
            )
            soft = RelaxedBernoulli(
                temperature=temperature, logits=boundary_logits
            ).rsample()
        hard = (soft > 0.5).float()
        boundaries = hard - soft.detach() + soft  # straight-through estimator

        valid_mask = (
            torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        ).float()
        # Padded positions must never register as boundaries: downsample/upsample
        # size their pooled sequence off boundaries.sum(-1).max() with no separate
        # masking, so a phantom boundary in the padding region would inflate the
        # segment count. Multiplying AFTER the straight-through construction keeps
        # gradient flow to boundary_logits for real positions while zeroing the
        # forward value in pad (same fix as magnet/model.py).
        boundaries = boundaries * valid_mask

        pooled = downsample(boundaries, h_pre, self.null_group)  # (B, S+1, D)
        mid_out = self.mid(pooled)  # (B, S+1, D)
        upsampled = upsample(boundaries, mid_out) + h_pre  # residual: byte-local
        # detail mean-pooling discarded. No key_padding_mask needed for `self.mid`
        # despite varying pooled lengths -- see module docstring's key_padding_mask
        # judgment call (causal masking alone suffices).

        post_out = self.post(upsampled)
        logits = self.byte_head(post_out)

        return {
            "logits": logits,
            "boundaries": boundaries,
            "valid_mask": valid_mask,
        }
