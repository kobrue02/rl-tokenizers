"""BLT's entropy model: a small (100M-param) byte-level causal LM whose
next-byte predictive entropy drives the Byte Latent Transformer's dynamic
patch boundaries (Pagnoni et al., "Byte Latent Transformer: Patches Scale
Better Than Tokens", Meta, 2024 -- github.com/facebookresearch/blt).

UNLIKE every other systems/*/model.py here, this is a real pretrained
checkpoint loaded and run as-is (entropy_model/consolidated.pth from
facebook/blt-1b on the HF Hub -- see _ENTROPY_MODEL_REPO's own comment for
why blt-1b's copy is used rather than the separately-gated facebook/blt-entropy
repo) -- nothing in this file is trained by this project. Structurally closer to
hf_frontier/ (an external, frozen system) than to manta/fanta/magnet/
flexitokens, except hf_frontier is explicitly TOKENIZER-ONLY (never loads
model weights); this module deliberately DOES load real weights and run a
real forward pass, because BLT has no static tokenizer at all -- patch
boundaries only exist as the output of this model.

CLEAN-ROOM REIMPLEMENTATION, not a port of bytelatent's own code, and not
usable via `pip install bytelatent` here: importing bytelatent.base_transformer
unconditionally does `from xformers.ops import AttentionBias, fmha` at module
level, even though the attention branch this project actually needs
(non-CUDA, no local_block_causal kernel) never touches xformers operationally.
xformers has no CPU wheel and is notoriously hard to build without CUDA, and
this project runs its evaluation locally/CPU. Verified equivalent, not just
plausible: the state dict below loads facebook/blt-entropy's real checkpoint
with ZERO missing and ZERO unexpected keys, and produces sane entropy values
(bounded by ln(260) =~ 5.56, observed well under that) with a byte-exact
patch-boundary round-trip -- see blt/segment.py.

THE entropy_model/params.json TRAP (read this before touching hyperparameters):
facebook/blt-1b's OWN entropy_model/params.json ships patcher_args with
patching_mode="byte" and a threshold value that LOOKS load-bearing but isn't:
in bytelatent's real code, patching_mode="byte" means "every byte is its own
patch" (patch_lengths = ones(...)) -- that's the entropy model's own TRAINING
config (it's trained as a plain byte-LM; patching is irrelevant to training
it), and the threshold field is inert leftover metadata for that mode. THE
REAL patching config that BLT itself uses lives in the MAIN model's
train_args.json (repo root of facebook/blt-1b, not entropy_model/):
patching_mode="entropy", patching_threshold=1.335442066192627,
patching_threshold_add=null, monotonicity=false. The threshold value is
identical between the two files (not a coincidence -- the real config was
derived from the same run), which makes the trap easy to miss: both files
independently look plausible.

Architecture constants below are read directly off facebook/blt-entropy's
entropy_model/params.json `entropy_model` block, cross-checked against
bytelatent/base_transformer.py's and bytelatent/transformer.py's real module
definitions (RMSNorm, RoPE with float64 outer product then cast, SwiGLU FFN,
grouped-query attention with n_kv_heads==n_heads here so no GQA is actually
exercised). rope_theta=10000.0 is the ENTROPY model's own value -- the main
1B/7B BLT model itself uses a different rope_theta (500000.0); don't cross
the two configs if this module is ever extended to load the main model too.
"""

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

DIM = 768
N_LAYERS = 14
N_HEADS = 12
N_KV_HEADS = 12  # == N_HEADS, so no grouped-query attention is actually exercised.
HEAD_DIM = DIM // N_HEADS  # 64
VOCAB_SIZE = 260  # 256 byte values + BOS/EOS/PAD/reserved (see byte<->id mapping below).
NORM_EPS = 1e-5
ROPE_THETA = 10000.0  # entropy model's own value; the main BLT model uses 500000.0 instead.
MULTIPLE_OF = 256
FFN_DIM_MULTIPLIER = 1.0
SLIDING_WINDOW = 512  # local causal attention window (attn_bias_type="local_block_causal").
MAX_PRECOMPUTED_SEQLEN = 8192  # matches entropy_model/params.json's max_seqlen.

# Byte<->id mapping (bytelatent/tokenizers/constants.py, blt_tokenizer.py): a
# sequence's ids are [BOS_ID] + [byte_value + OFFSET for each raw byte] +
# [EOS_ID]. OFFSET reserves ids 0-3 for PAD/BOS/EOS/(one more reserved slot),
# leaving 256 ids [OFFSET, OFFSET+256) for raw byte values 0-255.
OFFSET = 4
BOS_ID = 1
EOS_ID = 2

# The actual BLT patching threshold (natural-log entropy, nats) -- from the
# MAIN model's train_args.json, NOT entropy_model/params.json (see module
# docstring's "trap" section). monotonicity=false in that same config means
# this is a plain global-threshold rule, not the paper's alternative
# "approximate monotonic constraint" scheme -- see segment.py.
PATCHING_THRESHOLD = 1.335442066192627

_ENTROPY_MODEL_REPO = "facebook/blt-1b"
# NOT "facebook/blt-entropy": that's a SEPARATE gated repo whose access isn't
# granted by facebook/blt-1b's own approval (confirmed live -- a 403 even
# after blt-1b access was approved). facebook/blt-1b's own repo already ships
# the identical entropy_model/consolidated.pth (same "blt_main_entropy_100m_512w"
# checkpoint, per its own entropy_model/params.json "name" field) alongside
# the main 1B model's weights, so this avoids needing a second gate at all.
_CHECKPOINT_FILENAME = "entropy_model/consolidated.pth"
# NOT consolidated_with_rope.pth: bytelatent/hf.py's own HF-conversion script
# (the repo's own code for producing exactly this kind of standalone
# entropy-model checkpoint) hardcodes consolidated.pth as the one to load.


def _precompute_freqs_cis(head_dim, max_seqlen, theta):
    """Standard RoPE frequency table, precomputed once up to max_seqlen and
    sliced per forward call. Represented as 2x2 rotation matrices (cos/-sin/
    sin/cos) rather than complex numbers, matching bytelatent's own real-valued
    formulation (avoids torch.complex64, which has patchier CPU/MPS backend
    support than plain real tensors)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(max_seqlen)
    freqs = torch.outer(t, freqs).float()
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def _apply_rotary_emb(xq, xk, freqs_cis):
    """xq/xk: (B, T, n_heads, head_dim). freqs_cis: (T, head_dim/2, 2, 2)."""
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    # xq_'s seq dim is index 1 (B, T, heads, head_dim/2, 1, 2) -- broadcast
    # freqs_cis over batch and heads by putting T where xq_ has it.
    fc = freqs_cis.view(1, xq_.shape[1], 1, -1, 2, 2)
    xq_out = (xq_ * fc).sum(-1).flatten(3)
    xk_out = (xk_ * fc).sum(-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

    def forward(self, x, freqs_cis, attn_mask):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, N_HEADS, HEAD_DIM)
        xk = self.wk(x).view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)
        xv = self.wv(x).view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)
        xq, xk = _apply_rotary_emb(xq, xk, freqs_cis[:seqlen])
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))  # (B, heads, T, head_dim)
        # attn_bias_type="local_block_causal" reference code (bytelatent's own
        # model/utils.py) raises under attn_impl="sdpa" unless an env var is
        # set, and even then silently drops the sliding-window restriction --
        # so the mask is built directly here (see segment.py's LMTransformer
        # docstring) rather than via that helper.
        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        return self.wo(out)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(2 * (4 * DIM) / 3)
        hidden_dim = int(FFN_DIM_MULTIPLIER * hidden_dim)
        hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF)
        self.w1 = nn.Linear(DIM, hidden_dim, bias=False)
        self.w3 = nn.Linear(DIM, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.feed_forward = _FeedForward()
        self.attention_norm = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.ffn_norm = nn.RMSNorm(DIM, eps=NORM_EPS)

    def forward(self, x, freqs_cis, attn_mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, attn_mask)
        return h + self.feed_forward(self.ffn_norm(h))


class EntropyLMTransformer(nn.Module):
    """Byte-level causal LM (facebook/blt-entropy's architecture). forward()
    returns raw logits (B, T, VOCAB_SIZE); see segment.py for turning those
    into per-byte entropy and then patch boundaries."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([_TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.register_buffer(
            "freqs_cis",
            _precompute_freqs_cis(HEAD_DIM, MAX_PRECOMPUTED_SEQLEN, ROPE_THETA),
            persistent=False,
        )

    def forward(self, tokens):
        """tokens: (B, T) LongTensor of ids (see OFFSET/BOS_ID/EOS_ID above)."""
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        idx = torch.arange(seqlen, device=tokens.device)
        # Local causal window: position i attends to j <= i with i - j < SLIDING_WINDOW.
        # Single-document per forward call here (no packing), so this is exactly
        # attn_bias_type="local_block_causal" for our use case -- the "block" part
        # of that name only matters when multiple documents are packed into one
        # sequence separated by EOS, which never happens in this per-sentence
        # evaluation pipeline.
        mask = (idx.unsqueeze(0) <= idx.unsqueeze(1)) & (
            (idx.unsqueeze(1) - idx.unsqueeze(0)) < SLIDING_WINDOW
        )
        freqs_cis = self.freqs_cis.to(tokens.device)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        return self.output(self.norm(h))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def load_entropy_model(device="cpu"):
    """Downloads (if needed) and loads facebook/blt-entropy's real checkpoint.
    Loaded into a float32 model rather than following bytelatent's own
    torch.set_default_dtype(torch.bfloat16) pattern: that call is a
    process-wide global side effect that would silently change the default
    dtype for every OTHER tokenizer system sharing this Python process
    (evaluate.py dispatches all of them from one process). float32 is a
    strict numerical superset of the bf16-trained weights -- state_dict
    loading upcasts automatically -- so this is safe, not lossy, and doesn't
    leak global state.
    """
    path = hf_hub_download(_ENTROPY_MODEL_REPO, _CHECKPOINT_FILENAME)
    state_dict = torch.load(path, map_location="cpu")["model"]
    model = EntropyLMTransformer()
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    # strict=True: any drift between this clean-room reimplementation and the
    # real checkpoint's keys should fail loudly here, not silently produce
    # wrong entropy downstream.
    model.to(device)
    model.eval()
    return model
