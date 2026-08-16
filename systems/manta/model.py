"""MANTa: a gradient-based, boundary-free neural tokenizer.

Paper: Godey, Castagné, de la Clergerie & Sagot, "MANTa: Efficient
Gradient-Based Tokenization for Robust End-to-End Language Modeling"
(Findings of EMNLP 2022, aclanthology.org/2022.findings-emnlp.207).

Core idea (Eq. 1): instead of a hard, non-differentiable segmentation step
(BPE, or fairtok's sampled-Bernoulli boundaries), predict a per-byte
"frontier" probability p_i and treat byte i's block index as the cumulative
count of frontier events up to i -- a sum of Bernoulli(p_k) variables
(Poisson-Binomial), approximated by a moment-matched Gaussian. This makes
P(byte i in block b) closed-form and differentiable for every (i, b) pair: a
soft (T, num_blocks) assignment matrix instead of a discrete cut, so the LM
loss backprops straight through the segmentation. Everything else
(embeddings, block-level GRU, upsampling) is standard plumbing around this.

Deliberately scaled down vs. the paper, to sit next to fairtok/ as a
CPU-trainable comparison baseline rather than reproducing the paper's
~200M-param T5-scale model:

  1. No Longformer. Frontier predictor is a from-scratch bidirectional
     sliding-window attention (`SlidingWindowAttention`, +/- `window`) to
     keep zero dependency on `transformers`, matching fairtok. Unlike
     fairtok's causal GRU policy, this looks both directions since MANTa's
     boundaries are a deterministic function of the whole sequence (nothing
     downstream needs autoregressive sampling). Implemented as a full (T,T)
     score matrix masked to the local band rather than a true banded
     kernel -- fine at this project's sequence lengths, would need
     revisiting for long documents.

  2. No T5 span-corruption objective; trained with plain next-byte
     cross-entropy instead, so baselines compare tokenization mechanism
     alone, not pretraining objective.

     EMPIRICAL CONSEQUENCE: the paper's encoder never sees the spans the
     decoder predicts, so its bidirectionality can't leak the target. Here
     it can -- byte i's assignment/hidden state is influenced by byte i+1,
     the very target the causal loss predicts. Confirmed live: average span
     length collapsed to ~1.06 bytes (~95% singletons) within ~10/80
     smoke-test steps even as loss kept falling, i.e. the model partly
     exploits the leak rather than learning to compress. Forcing the block
     GRU to be causal only reduced this to ~88% singletons, so the frontier
     predictor's own required bidirectionality (point 1) is the dominant
     leak. Reported as a finding, not patched: a bidirectional boundary-free
     tokenizer + naive causal LM loss has a structural incentive toward
     degenerate segmentation.

  3. No auxiliary loss -- true to the paper: pure LM cross-entropy, no
     rate/entropy/fairness term (unlike fairtok, MAGNET, FlexiTokens).

  4. Pooling simplification: skips the paper's depthwise conv before
     pooling, pools raw byte embeddings directly (weighted-average matmul
     against the assignment matrix). Accepted simplification; costs some
     local-context sharpening.

  5. Block-level layers use a GRU, not a second transformer stack --
     matches fairtok's house style and avoids a redundant attention
     implementation. Order still matters, so this is a reasonable swap.

  6. Upsampling is soft (assignment_matrix @ block_hidden) throughout the
     forward pass, keeping it fully differentiable. Hard-argmax
     discretization into actual token boundaries is a separate,
     inference-only step in manta/segment.py, never used in the loss path.

  7. Numerical floor on the Gaussian variance (`_MIN_VARIANCE`): sigma_i^2
     can approach 0 as frontier probabilities saturate, blowing up the
     log-density. Pure numerical guard, not part of the paper's Eq. 1.

  8. Scale: dim~64, small attention window, single-layer bidirectional GRU
     -- sized to match fairtok's own BytePolicy, not the paper's ~200M
     params.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Floor on the Poisson-Binomial-approximating Gaussian's variance (see point 7
# above). Deliberately tiny -- only matters once a frontier probability
# saturates near 0/1, where it's the difference between a valid soft
# assignment and a NaN from dividing by ~0 variance.
_MIN_VARIANCE = 1e-3


def sinusoidal_positional_encoding(length, dim, device):
    """Standard Transformer (Vaswani et al. 2017) sinusoidal position encoding.

    Content-only dot-product attention can't distinguish left from right, so
    this is added to the frontier predictor's input only (not to the raw
    byte embeddings used for pooling) -- only the frontier decision needs
    position info.
    """
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(
        1
    )  # (T, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )  # (ceil(dim/2),)
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    # dim may be odd; cos gets whatever's left after the sin columns above claimed
    # ceil(dim/2) of them, so slice div_term down to however many cos columns exist.
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe  # (T, dim)


class SlidingWindowAttention(nn.Module):
    """Bidirectional local-window multi-head self-attention (module docstring,
    point 1: from-scratch, no `transformers` Longformer). Position i attends
    only to j with |i - j| <= window, both directions. Implemented by masking
    a full (T, T) score matrix rather than a true banded kernel (see module
    docstring for the complexity tradeoff).
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
        qkv = (
            self.qkv(x)
            .view(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, H, T, T)

        idx = torch.arange(T, device=x.device)
        # allowed[i, j] = True iff j is within `window` positions of i, in EITHER
        # direction -- the "bidirectional local-window" part of the design.
        outside_band = (
            idx.unsqueeze(0) - idx.unsqueeze(1)
        ).abs() > self.window  # (T, T)
        disallow = outside_band.unsqueeze(0).unsqueeze(
            0
        )  # (1, 1, T, T), broadcasts over B, H
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
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class FrontierPredictor(nn.Module):
    """Stack of FrontierLayers producing one boundary logit per byte: how
    likely a new block starts there, feeding the Gaussian approximation in
    MantaModel.forward."""

    def __init__(self, dim, num_heads, window, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [FrontierLayer(dim, num_heads, window) for _ in range(num_layers)]
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.head(x).squeeze(-1)  # (B, T) raw logits, pre-sigmoid


@dataclasses.dataclass
class MantaOutput:
    """Everything downstream code needs off one forward pass. `assignment` is
    kept (not discarded after pooling/upsampling) since segment.py's
    discretization and train.py's token-frequency tracking both read it
    directly, avoiding a second forward pass."""

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
        # How many stddevs past the sequence-end mean block index to keep as
        # candidate blocks (mu_L + max_extra_sigma*sigma_L). A config knob since
        # wider/narrower trades off clipping true tail mass vs. extra
        # near-zero-weight block columns the GRU has to process.
        self.max_extra_sigma = max_extra_sigma

        self.byte_embed = nn.Embedding(256, dim)
        self.frontier = FrontierPredictor(
            dim, num_frontier_heads, window, num_frontier_layers
        )
        # Bidirectional: matches the paper's non-causal, encoder-style
        # block-level layers (module docstring point 5).
        self.block_rnn = nn.GRU(
            dim,
            block_hidden_size,
            num_layers=num_block_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Project bidirectional GRU's 2*block_hidden_size output back to `dim`.
        self.block_proj = nn.Linear(block_hidden_size * 2, dim)
        self.output_head = nn.Linear(dim, 256)

    def forward(self, byte_ids, lengths):
        """byte_ids: (B, T) LongTensor, right-padded with zeros past each
        sequence's real length. lengths: (B,) LongTensor of real lengths.
        Returns a MantaOutput.

        Assumes right-padding: mu/sigma below are left-to-right cumulative
        sums, so padding after position i can't leak into i's own mu_i/
        sigma_i -- left-padding would break that invariant (matches
        fairtok.policy.batched_sample_rollout's convention).
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        byte_emb = self.byte_embed(
            byte_ids
        )  # (B, T, dim) -- pooled on directly, no positional info

        position_idx = torch.arange(T, device=device)
        padding_mask = position_idx.unsqueeze(0) >= lengths.unsqueeze(
            1
        )  # (B, T), True = pad

        # Positional encoding only for the frontier predictor's input -- the raw
        # byte_emb used for pooling stays position-free.
        frontier_input = byte_emb + sinusoidal_positional_encoding(
            T, self.dim, device
        ).unsqueeze(0)
        frontier_logit = self.frontier(frontier_input, padding_mask)  # (B, T)
        p = torch.sigmoid(frontier_logit)
        # Belt-and-suspenders: right-padding + left-to-right cumsum already keep
        # pad positions from affecting real mu/sigma, but zero them here too so p
        # itself stays clean at pad positions.
        p = p.masked_fill(padding_mask, 0.0)

        # --- Eq. 1: Gaussian approximation to the Poisson-Binomial block index ---
        # block_index(i) := sum_{k<=i} Bernoulli(p_k); mu/sigma are its exact
        # mean/variance (moment-matched Gaussian).
        mu = torch.cumsum(p, dim=1)  # (B, T)
        variance = torch.cumsum(p * (1 - p), dim=1).clamp_min(_MIN_VARIANCE)
        sigma = torch.sqrt(variance)

        # Truncate the candidate block range using each sequence's LAST real
        # (mu, sigma), then take the batch max so every sequence gets the same
        # num_blocks (needed to batch the GRU below). Shorter/lower-mu sequences
        # just get some trailing near-zero-weight block columns -- harmless.
        last_idx = (lengths - 1).clamp_min(0)
        mu_L = mu.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B,)
        sigma_L = sigma.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B,)
        num_blocks_per_seq = (
            torch.ceil(mu_L + self.max_extra_sigma * sigma_L).long() + 1
        )
        num_blocks = int(num_blocks_per_seq.max().clamp_min(1).item())

        block_idx = (
            torch.arange(num_blocks, device=device).view(1, 1, num_blocks).float()
        )  # (1,1,nb)
        mu_exp = mu.unsqueeze(-1)  # (B, T, 1)
        sigma_exp = sigma.unsqueeze(-1)  # (B, T, 1)
        # Log-density up to the per-i normalizer, which softmax over b cancels.
        log_density = -((block_idx - mu_exp) ** 2) / (2 * sigma_exp**2)
        assignment = torch.softmax(
            log_density, dim=-1
        )  # (B, T, num_blocks), soft P(byte i in block b)
        # softmax alone only guarantees rows sum to 1, not that pad rows are
        # zero, so mask padded byte rows explicitly.
        assignment = assignment.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # --- Pooling: weighted average of raw byte embeddings (batched matmul,
        # module docstring point 4) ---
        weighted_sum = torch.bmm(
            assignment.transpose(1, 2), byte_emb
        )  # (B, num_blocks, dim)
        block_weight = assignment.sum(dim=1, keepdim=True).transpose(
            1, 2
        )  # (B, num_blocks, 1)
        # Epsilon guards ~0/~0 -> NaN for unused truncation-tail block columns;
        # numerator is equally tiny there so the quotient just goes to ~0.
        pooled = weighted_sum / (block_weight + 1e-8)  # (B, num_blocks, dim)

        # --- Block-level context (module docstring, point 5) ---
        block_hidden, _ = self.block_rnn(pooled)  # (B, num_blocks, 2*block_hidden_size)
        block_hidden = self.block_proj(block_hidden)  # (B, num_blocks, dim)

        # --- Upsample: soft weighted combination over blocks (same assignment
        # matrix pooling used), keeping the path differentiable end to end
        # (module docstring point 6). Hard-argmax discretization lives
        # separately in manta/segment.py. ---
        byte_hidden = torch.bmm(assignment, block_hidden)  # (B, T, dim)

        logits = self.output_head(byte_hidden)  # (B, T, 256)
        return MantaOutput(
            logits=logits, assignment=assignment, frontier_prob=p, mu=mu, sigma=sigma
        )

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def next_byte_loss(byte_ids, lengths, logits):
    """Plain next-byte cross-entropy -- the entire loss, no auxiliary terms
    (module docstring point 3). Returns (loss, num_valid_positions,
    num_correct) so callers can report accuracy/bits-per-byte without a
    second pass. Position t predicts byte_ids[:, t+1]; each sequence's last
    real byte has no next byte, so it's excluded via `valid_mask`.
    """
    B, T = byte_ids.shape
    device = byte_ids.device
    targets = torch.zeros_like(byte_ids)
    targets[:, :-1] = byte_ids[:, 1:]

    position_idx = torch.arange(T, device=device)
    valid_mask = position_idx.unsqueeze(0) < (lengths - 1).unsqueeze(1)  # (B, T)

    per_position_loss = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )  # (B, T)
    num_valid = valid_mask.sum().clamp_min(1)
    loss = (per_position_loss * valid_mask).sum() / num_valid

    predictions = logits.argmax(dim=-1)
    num_correct = ((predictions == targets) & valid_mask).sum()
    return loss, int(num_valid.item()), int(num_correct.item())
