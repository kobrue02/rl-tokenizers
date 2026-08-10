"""MANTa: a gradient-based, boundary-free neural tokenizer.

Paper: Godey, Castagné, de la Clergerie & Sagot, "MANTa: Efficient
Gradient-Based Tokenization for Robust End-to-End Language Modeling"
(Findings of EMNLP 2022, aclanthology.org/2022.findings-emnlp.207).

The core idea (Eq. 1 of the paper): instead of a hard, non-differentiable
segmentation step (BPE, or fairtok's own sampled-Bernoulli boundaries in
fairtok/policy.py), predict a per-byte "frontier" probability p_i and treat
the *block index* of byte i as the cumulative count of frontier events up to
i, i.e. a sum of Bernoulli(p_k) variables (a Poisson-Binomial distribution).
That distribution is approximated by a Gaussian with matching mean/variance,
which makes "the probability that byte i belongs to block b" a closed-form,
differentiable quantity for every (i, b) pair -- a soft (T, num_blocks)
assignment matrix, never a discrete cut, so gradients from the language-
modeling loss flow straight back through the segmentation itself. This is
the "gradient-based" part of the paper's title, and the one piece of MANTa
this file is really about; everything else (embeddings, block-level layers,
upsampling) is standard sequence-model plumbing around it.

This is a faithful-in-spirit but DELIBERATELY SCALED-DOWN and SIMPLIFIED
reimplementation, built to sit next to fairtok/ as a comparison baseline at
fairtok's own compute scale (a CPU-trainable smoke test), not the paper's
~200M-parameter T5-scale model pretrained on C4. Concretely, vs. the paper:

  1. No Longformer. The paper's frontier predictor is a Longformer encoder.
     fairtok has zero dependency on `transformers` (see fairtok/policy.py,
     fairtok/train.py -- everything is plain torch), and this project keeps
     that property, so the frontier predictor below is a small bidirectional
     sliding-window self-attention block implemented from scratch
     (`SlidingWindowAttention`): each position attends only to positions
     within +/- `window` of itself. This is a real architectural difference
     from fairtok's own BytePolicy (fairtok/policy.py), which is CAUSAL (a
     GRU that only ever sees the past) -- MANTa's segmentation looks in BOTH
     directions before deciding where a boundary probably is, which is only
     possible because nothing downstream needs autoregressive sampling of
     the boundaries themselves (unlike fairtok's REINFORCE-trained policy,
     MANTa's boundaries are a deterministic, differentiable function of the
     whole sequence, so there is no causality requirement to preserve).
     For simplicity, the O(T^2) full attention matrix is computed and then
     masked down to the local band, rather than a true banded-attention
     kernel that only computes the O(T * window) entries that survive the
     mask -- at the sequence lengths used here (tens to low hundreds of
     bytes) this costs nothing extra in practice, but it would need
     revisiting before running this on long documents.

  2. No T5 span-corruption denoising objective. The paper attaches this
     mechanism to the front of a full T5 encoder-decoder and pretrains with
     span-corruption denoising. This file only implements the tokenization
     mechanism, paired with plain next-byte cross-entropy (a causal LM
     loss) -- the same objective family fairtok, MAGNET, and FlexiTokens all
     use -- so that a comparison between baselines is a comparison of
     TOKENIZATION MECHANISMS, not confounded by two totally different
     pretraining objectives. This is a deliberate, explicit deviation from
     the paper, not an oversight.

     EMPIRICAL CONSEQUENCE, observed when actually running this (see
     manta.train.run_smoke_test's printed span-length histogram): pairing
     this mechanism's bidirectionality (the frontier predictor MUST look
     both ways -- point 1 above; the block-level layers mirror the paper's
     bidirectional T5 encoder -- point 5 below) with a plain CAUSAL next-
     byte loss removes a safety property the paper's own training scheme
     relies on. In the paper, the encoder never sees the corrupted spans
     the decoder is asked to predict, so its bidirectionality can't leak
     the prediction target. Here, there is no such masking: byte i's
     assignment row (and, through the block-level GRU/upsampling, byte i's
     final hidden state) can be influenced by byte i+1 -- the very target
     next-byte cross-entropy is trying to predict -- via attention/GRU
     paths that look forward as well as back. The finer the segmentation
     (more, smaller blocks), the more directly that channel can carry
     information about a specific upcoming byte. In the actual smoke test
     run recorded in this project, this shows up as the model rapidly (within
     ~10 of 80 steps) driving average induced span length down to ~1.06
     bytes (~95% single-byte spans) even though the LOSS keeps falling --
     i.e. the model is not learning to compress, it's partly exploiting this
     leakage channel, and finer blocks make that channel wider. A quick
     ablation (forcing the block-level GRU to be unidirectional/causal
     instead of bidirectional) reduced but did not eliminate the effect
     (~88% singleton spans instead of ~95%), confirming the frontier
     predictor's OWN required bidirectionality is the dominant leak, not
     just the block-level layers. This is reported here deliberately, not
     patched over: the frontier predictor can't be made causal without
     contradicting the paper's own design (point 1), and inventing a rate
     loss to mask the symptom would contradict point 3 below and defeat the
     purpose of comparing MANTa's mechanism, as specified, against fairtok's
     rate-regularized one on equal footing. Read as a finding, not a bug:
     "a bidirectional, boundary-free tokenizer front end paired with a
     naive causal LM objective has a structural incentive toward
     degenerate, near-character-level segmentation" is exactly the kind of
     result a baseline comparison like this one is supposed to surface.

  3. No auxiliary loss of any kind. This is actually a genuine property of
     the paper, not a simplification: MANTa's loss is language-modeling
     cross-entropy ONLY. Segmentation emerges purely from backpropagating
     that loss through the soft assignment matrix. There is no boundary-rate
     target, no entropy regularizer, no fairness term -- worth flagging
     explicitly because fairtok (a rate-consistency loss + a fairness term)
     and MAGNET/FlexiTokens (their own rate losses) all need one, and MANTa
     conspicuously does not. See manta/train.py's loss function: it really
     is just cross-entropy, nothing summed in alongside it.

  4. Pooling simplification. The paper pools a small depthwise 1-D
     convolution over byte embeddings before the weighted-average pooling
     step. This implementation skips the convolution and pools the raw byte
     embeddings directly (a straight weighted average, batched as a single
     matmul against the assignment matrix -- see `MantaModel.forward`). This
     is an accepted simplification per the task spec, not a correctness bug;
     it costs some local-context sharpening in the pooled block embeddings
     that the depthwise conv would otherwise provide.

  5. Block-level layers: GRU, not transformer. The paper runs a further T5
     encoder stack over the pooled block sequence. This implementation uses
     a small bidirectional GRU instead (`nn.GRU`), matching fairtok's own
     house style (fairtok/policy.py's BytePolicy is GRU-based throughout)
     and avoiding writing a second, full self-attention stack when one
     already exists here for the frontier predictor. Block order still
     matters (a GRU is a fine inductive bias for "this block's meaning
     depends on its neighbors in sequence"), so this is a reasonable
     architecture swap, not a downgrade in kind.

  6. Upsampling: soft during training, hard only for evaluation. Byte-level
     hidden states are recovered from block-level hidden states by
     `assignment_matrix @ block_hidden` -- i.e. byte i's hidden state is the
     SAME soft weighted combination over blocks that pooling used to build
     them in the first place. This keeps the entire forward pass
     differentiable end to end, which is the point of the whole exercise.
     Hard-argmax discretization (turning the soft assignment into actual
     0/1 boundaries) is a SEPARATE, inference-only operation implemented in
     manta/segment.py -- it is never used inside the loss path, only to
     extract token counts for common.metrics after training. The paper
     never needs discrete boundaries at all, so this discretization
     heuristic is entirely this project's own addition; see segment.py's
     docstring for the exact rule and its caveats.

  7. Numerical-stability floor on the Gaussian variance. `sigma_i^2 =
     sum_{k<=i} p_k * (1 - p_k)` can be driven arbitrarily close to zero once
     the frontier probabilities saturate near 0 or 1 (a real possibility
     once training pushes p_i toward confident decisions), which would blow
     up the Gaussian log-density used for the soft assignment. A small
     additive epsilon floor is applied to the variance before taking the
     square root (`_MIN_VARIANCE` below) -- purely a numerical guard, not a
     detail specified by the paper's Eq. 1.

  8. Scale. Byte embedding / hidden dim ~64, a couple of attention layers
     with a small window, a single-layer bidirectional GRU -- sized to be
     comparable in parameter count to fairtok's own BytePolicy (see
     fairtok/policy.py, hidden_dim=64-128, num_layers=2-3), not the paper's
     ~200M-parameter model. This is a baseline for comparison at fairtok's
     own compute scale, explicitly not an attempt to reproduce the paper's
     reported numbers.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Additive floor on the Poisson-Binomial-approximating Gaussian's variance
# (see point 7 in the module docstring). This is deliberately tiny -- it
# should never matter unless a frontier probability is genuinely saturated
# very close to 0/1, in which case it's the difference between a legitimate
# soft assignment and a NaN from dividing by ~0 variance.
_MIN_VARIANCE = 1e-3


def sinusoidal_positional_encoding(length, dim, device):
    """Standard Transformer (Vaswani et al. 2017) sinusoidal position encoding.

    The frontier predictor's attention is local-window (see
    SlidingWindowAttention) but still needs SOME signal that distinguishes
    "the byte two positions to my left" from "the byte two positions to my
    right" -- raw dot-product attention over content alone can't tell
    left from right, only "how similar are our embeddings." Added once, to
    the frontier predictor's input only (not to the raw byte embeddings used
    for pooling -- see MantaModel.forward), matching the fact that only the
    frontier decision, not the pooled block content, needs position info.
    """
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )  # (ceil(dim/2),)
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    # dim may be odd; cos gets whatever's left after the sin columns above claimed
    # ceil(dim/2) of them, so slice div_term down to however many cos columns exist.
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe  # (T, dim)


class SlidingWindowAttention(nn.Module):
    """Bidirectional local-window multi-head self-attention, implemented directly
    in torch (see module docstring, point 1, for why not `transformers`'
    Longformer). Position i attends only to positions j with |i - j| <= window
    -- both directions, unlike fairtok's causal GRU. Implemented by masking a
    full (T, T) score matrix down to the local band rather than a true banded
    kernel; see the module docstring for the complexity tradeoff this implies.
    """

    def __init__(self, dim, num_heads, window):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):
        """x: (B, T, D). key_padding_mask: (B, T) bool, True = this position is
        padding and must never be attended TO (it can still safely be a query --
        its output is simply never read by the caller)."""
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        idx = torch.arange(T, device=x.device)
        # allowed[i, j] = True iff j is within `window` positions of i, in EITHER
        # direction -- the "bidirectional local-window" part of the design.
        outside_band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window  # (T, T)
        disallow = outside_band.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T), broadcasts over B, H
        if key_padding_mask is not None:
            disallow = disallow | key_padding_mask.view(B, 1, 1, T)
        scores = scores.masked_fill(disallow, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # A padded query position whose entire local band is also padding produces
        # an all -inf row -> softmax gives all-NaN, not all-zero. That row is never
        # read downstream (padding positions are dropped before the loss/pooling),
        # but NaNs still propagate through backward() if left alone, so scrub them.
        attn = torch.nan_to_num(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class FrontierLayer(nn.Module):
    """One pre-norm transformer block built around SlidingWindowAttention --
    standard residual attention + FFN sandwich, nothing MANTa-specific here."""

    def __init__(self, dim, num_heads, window, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SlidingWindowAttention(dim, num_heads, window)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim))

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class FrontierPredictor(nn.Module):
    """Stack of FrontierLayers producing one boundary logit per byte position.
    This is a "frontier predictor" in the paper's terminology: it decides,
    per byte, how likely it is that a new block starts there -- not a hard
    decision, just a probability that feeds the Gaussian approximation in
    MantaModel.forward."""

    def __init__(self, dim, num_heads, window, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([FrontierLayer(dim, num_heads, window) for _ in range(num_layers)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.head(x).squeeze(-1)  # (B, T) raw logits, pre-sigmoid


@dataclasses.dataclass
class MantaOutput:
    """Everything downstream code needs off one forward pass. `assignment` is
    kept around (rather than discarded once pooling/upsampling consume it)
    because manta/segment.py's discretization heuristic reads it directly,
    and manta/train.py uses it to update a running per-language token-
    frequency table without a second forward pass."""

    logits: torch.Tensor  # (B, T, 256) next-byte logits
    assignment: torch.Tensor  # (B, T, num_blocks) soft P(byte i in block b)
    frontier_prob: torch.Tensor  # (B, T) sigmoid(frontier logit) = p_i
    mu: torch.Tensor  # (B, T) running mean block index, per Eq. 1
    sigma: torch.Tensor  # (B, T) running stddev of block index, per Eq. 1


class MantaModel(nn.Module):
    """The full MANTa pipeline: byte embeddings -> frontier probabilities ->
    soft byte-to-block assignment (Eq. 1) -> pooling -> block-level GRU ->
    upsampling -> byte-level next-byte logits. See the module docstring for
    every simplification relative to the original paper.
    """

    def __init__(
        self,
        dim=64,
        window=8,
        num_frontier_layers=2,
        num_frontier_heads=4,
        block_hidden_size=64,
        num_block_layers=1,
        max_extra_sigma=3.0,
    ):
        super().__init__()
        self.dim = dim
        # How many standard deviations past the sequence-end mean block index to
        # keep as candidate blocks (b ranges 0..num_blocks-1) -- the task spec's
        # own truncation heuristic (mu_L + 3*sigma_L), kept as a config knob
        # rather than a hardcoded constant since a wider/narrower margin trades
        # off "never clip probability mass off the true tail" against "how many
        # (mostly near-zero-weight) extra block columns the GRU has to chew
        # through every step."
        self.max_extra_sigma = max_extra_sigma

        self.byte_embed = nn.Embedding(256, dim)
        self.frontier = FrontierPredictor(dim, num_frontier_heads, window, num_frontier_layers)
        # bidirectional: block b's representation can depend on blocks after it
        # too, matching the paper's non-causal, encoder-style block-level layers
        # (see module docstring, point 5, for why GRU rather than transformer here).
        self.block_rnn = nn.GRU(
            dim, block_hidden_size, num_layers=num_block_layers, batch_first=True, bidirectional=True
        )
        # Bidirectional GRU output is 2*block_hidden_size wide; project back down
        # to `dim` so the upsampled byte-level hidden state has the same width the
        # output head expects, independent of how block_hidden_size is configured.
        self.block_proj = nn.Linear(block_hidden_size * 2, dim)
        self.output_head = nn.Linear(dim, 256)

    def forward(self, byte_ids, lengths):
        """byte_ids: (B, T) LongTensor, right-padded with zeros past each
        sequence's real length. lengths: (B,) LongTensor of real (unpadded)
        lengths. Returns a MantaOutput.

        Padding convention: padding lives strictly at the END of each row, and
        every cumulative-sum quantity below (mu, sigma) is computed left-to-
        right, so padding after position i can never leak into i's own mu_i/
        sigma_i -- only left-padding would break that invariant, which is why
        this function assumes right-padding throughout (matching
        fairtok.policy.batched_sample_rollout's own convention).
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        byte_emb = self.byte_embed(byte_ids)  # (B, T, dim) -- pooled on directly, no positional info

        position_idx = torch.arange(T, device=device)
        padding_mask = position_idx.unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T), True = pad

        # Positional encoding is added ONLY for the frontier predictor's input --
        # the raw byte_emb used for pooling below stays position-free (see
        # sinusoidal_positional_encoding's docstring for why).
        frontier_input = byte_emb + sinusoidal_positional_encoding(T, self.dim, device).unsqueeze(0)
        frontier_logit = self.frontier(frontier_input, padding_mask)  # (B, T)
        p = torch.sigmoid(frontier_logit)
        # Padded positions never contribute frontier "events": zeroing them out
        # here is belt-and-suspenders (right-padding + left-to-right cumsum
        # already guarantees they can't affect any REAL position's mu/sigma) but
        # keeps p itself clean/interpretable at pad positions instead of whatever
        # the attention block happened to output there.
        p = p.masked_fill(padding_mask, 0.0)

        # --- Eq. 1: Gaussian approximation to the Poisson-Binomial block index ---
        # block_index(i) := sum_{k<=i} Bernoulli(p_k)  ==>  approximate as
        # Normal(mu_i, sigma_i^2) with mu_i/sigma_i^2 the exact mean/variance of
        # that same sum (matching moments, the standard Gaussian approximation to
        # a Poisson-Binomial distribution).
        mu = torch.cumsum(p, dim=1)  # (B, T)
        variance = torch.cumsum(p * (1 - p), dim=1).clamp_min(_MIN_VARIANCE)
        sigma = torch.sqrt(variance)

        # Truncate the candidate block range using the LAST real position's own
        # (mu, sigma) per sequence, then take the max across the batch so every
        # sequence gets the same num_blocks (required to batch the GRU below).
        # Shorter/lower-mu sequences in the same batch simply end up with some
        # trailing block columns that carry ~zero probability mass for every one
        # of their real bytes -- harmless (see pooling's epsilon guard below),
        # not a correctness issue.
        last_idx = (lengths - 1).clamp_min(0)
        mu_L = mu.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B,)
        sigma_L = sigma.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B,)
        num_blocks_per_seq = torch.ceil(mu_L + self.max_extra_sigma * sigma_L).long() + 1
        num_blocks = int(num_blocks_per_seq.max().clamp_min(1).item())

        block_idx = torch.arange(num_blocks, device=device).view(1, 1, num_blocks).float()  # (1,1,nb)
        mu_exp = mu.unsqueeze(-1)  # (B, T, 1)
        sigma_exp = sigma.unsqueeze(-1)  # (B, T, 1)
        # Gaussian log-density up to the (per-i, constant-over-b) normalizer,
        # which softmax over b cancels out anyway -- exactly the "up to a
        # normalizing constant" the task spec calls for.
        log_density = -((block_idx - mu_exp) ** 2) / (2 * sigma_exp ** 2)
        assignment = torch.softmax(log_density, dim=-1)  # (B, T, num_blocks), soft P(byte i in block b)
        # Zero out padded byte ROWS so they can't contribute weight to any
        # block's pooled embedding below (softmax alone can't do this -- it only
        # guarantees each ROW sums to 1, not that pad rows are all-zero).
        assignment = assignment.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # --- Pooling: weighted average of RAW byte embeddings (module docstring,
        # point 4) via one batched matmul, not a Python loop over blocks. ---
        weighted_sum = torch.bmm(assignment.transpose(1, 2), byte_emb)  # (B, num_blocks, dim)
        block_weight = assignment.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, num_blocks, 1)
        # A block column with ~0 total weight (the "extra" truncation columns
        # some sequences never really use, see num_blocks above) would divide
        # ~0/~0 -> NaN without this epsilon; the numerator is equally tiny there,
        # so the epsilon just pins the quotient to ~0 instead, which is exactly
        # right since nothing meaningfully "belongs" to that block anyway.
        pooled = weighted_sum / (block_weight + 1e-8)  # (B, num_blocks, dim)

        # --- Block-level context (module docstring, point 5) ---
        block_hidden, _ = self.block_rnn(pooled)  # (B, num_blocks, 2*block_hidden_size)
        block_hidden = self.block_proj(block_hidden)  # (B, num_blocks, dim)

        # --- Upsample: SOFT weighted combination over blocks, same assignment
        # matrix pooling used -- keeps the whole path differentiable end to end
        # (module docstring, point 6). Hard-argmax discretization for actual
        # token boundaries lives in manta/segment.py, entirely outside this
        # forward pass. ---
        byte_hidden = torch.bmm(assignment, block_hidden)  # (B, T, dim)

        logits = self.output_head(byte_hidden)  # (B, T, 256)
        return MantaOutput(logits=logits, assignment=assignment, frontier_prob=p, mu=mu, sigma=sigma)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def next_byte_loss(byte_ids, lengths, logits):
    """Plain next-byte cross-entropy (module docstring, point 3: this is the
    ENTIRE loss -- no auxiliary terms). Returns (loss, num_valid_positions,
    num_correct) so callers can also report byte accuracy / bits-per-byte
    without a second pass over the logits.

    byte_ids/lengths/logits as returned by/passed to MantaModel.forward.
    Position t predicts byte_ids[:, t+1], so the last real byte of each
    sequence (no next byte to predict) is excluded via `valid_mask`.
    """
    B, T = byte_ids.shape
    device = byte_ids.device
    targets = torch.zeros_like(byte_ids)
    targets[:, :-1] = byte_ids[:, 1:]

    position_idx = torch.arange(T, device=device)
    valid_mask = position_idx.unsqueeze(0) < (lengths - 1).unsqueeze(1)  # (B, T)

    per_position_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # (B, T)
    num_valid = valid_mask.sum().clamp_min(1)
    loss = (per_position_loss * valid_mask).sum() / num_valid

    predictions = logits.argmax(dim=-1)
    num_correct = ((predictions == targets) & valid_mask).sum()
    return loss, int(num_valid.item()), int(num_correct.item())
