"""A LLaMA-style decoder-only transformer: RMSNorm, rotary position
embeddings (RoPE), SwiGLU-gated MLP, causal self-attention (optionally
grouped-query, for the largest presets) via torch's fused
scaled_dot_product_attention. This is the architecture essentially every
current open LLM (LLaMA, Mistral, Qwen, Gemma, ...) uses at every scale from
a few hundred million to well past 7B parameters -- see model_configs.py for
the named size presets built on top of this one architecture.

Unlike every tokenizer baseline in systems/, this is NOT a deliberately
scaled-down simplification of a paper's design -- it's meant to actually
pretrain a real language model, so it uses the same architectural choices a
real model would, just at whatever --model-size the caller picks (see
pretraining/train.py and model_configs.PRESETS).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: normalizes by RMS only (no mean
    subtraction, no bias) -- LLaMA's normalization of choice, cheaper than
    LayerNorm and empirically just as effective for this architecture
    family."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Computed in fp32 regardless of the ambient autocast dtype -- RMS
        # norm's own reciprocal-sqrt is exactly the kind of reduction that
        # loses meaningful precision in bf16 (small variance estimates get
        # rounded before the correction is even applied), a well-known
        # LLaMA-implementation detail, not a stylistic choice.
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(dtype)


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):
    """Returns (cos, sin), each (max_seq_len, head_dim) -- head_dim (not
    head_dim/2) because each half of a rotation pair gets the SAME
    cos/sin value repeated (see apply_rope's rotate_half convention, the
    GPT-NeoX/LLaMA style rather than the original RoFormer paper's
    interleaved-pairs style; the two are mathematically equivalent up to a
    fixed permutation of dimensions, LLaMA's own implementation uses this
    one)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )  # (head_dim/2,)
    t = torch.arange(max_seq_len, device=device).float()  # (T,)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return freqs.cos(), freqs.sin()


def _rotate_half(x):
    """x: (..., head_dim). Splits in half and swaps-with-negation -- the
    GPT-NeoX-style rotation companion to precompute_rope's cos/sin layout
    (see that function's docstring)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, num_heads, T, head_dim). cos, sin: (T, head_dim) sliced
    from precompute_rope's full table for this sequence's actual length.
    Returns (q_rotated, k_rotated), same shapes -- standard RoPE application,
    identical formula for q and k."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim), broadcasts over B, heads
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    """Causal self-attention with RoPE and optional grouped-query attention
    (num_kv_heads < num_heads -- see model_configs.py's "7b" preset). Uses
    F.scaled_dot_product_attention rather than a hand-rolled softmax(QK^T/
    sqrt(d))V -- lets PyTorch dispatch to a fused/flash-attention kernel on
    supported hardware, which matters a great deal at real pretraining
    scale (the hand-rolled version materializes a full (T, T) score matrix
    per head, exactly the O(T^2) memory cost that made systems.fanta's own
    dense-attention baseline OOM at far smaller scale than this -- see that
    package's max_seq_length field for the incident that taught this
    project that lesson)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_seq_len, rope_theta, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        assert T <= self.rope_cos.shape[0], (
            f"sequence length {T} exceeds this model's max_seq_len "
            f"{self.rope_cos.shape[0]} -- RoPE has no precomputed frequencies "
            "past that (raising here directly rather than letting this surface "
            "as a confusing shape mismatch inside apply_rope)"
        )
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:T].to(dtype=q.dtype, device=q.device)
        sin = self.rope_sin[:T].to(dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            # Grouped-query attention: each KV head is shared by
            # num_heads/num_kv_heads query heads -- repeat_interleave so
            # query head i uses kv head i // group_size, matching every
            # standard GQA implementation's head-grouping convention.
            group_size = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """LLaMA's gated MLP: down(silu(gate(x)) * up(x)) -- three matmuls
    (gate/up/down) instead of a plain GELU MLP's two, with intermediate_size
    shrunk accordingly (see model_configs._swiglu_intermediate_size) to keep
    total FLOPs comparable."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x = x + attn(norm1(x)); x = x + mlp(norm2(x))
    -- standard LLaMA-family block shape."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads_resolved(),
            cfg.max_seq_len,
            cfg.rope_theta,
            cfg.dropout,
        )
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    """Full decoder-only LM: token embedding -> N TransformerBlocks -> final
    RMSNorm -> output projection to vocab logits. vocab_size is passed
    separately from `cfg` (a model_configs.ModelConfig, which is purely
    architectural) since it comes from whichever tokenizer is in use, not
    from the chosen model size -- see model_configs.py's own docstring."""

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # Second, targeted pass: rescale the two per-block projections that
        # write directly into the residual stream (attn.o_proj, mlp.down_proj)
        # by 1/sqrt(2 * num_layers) -- GPT-2's own documented fix for the
        # residual stream's variance otherwise growing with depth at
        # initialization. Needs full dotted parameter names to target just
        # these two Linear layers, which self._init_weights (called via
        # nn.Module.apply, one undotted module at a time) can't see -- hence
        # a separate pass over self.named_parameters() here instead.
        residual_scale = 1.0 / math.sqrt(2 * cfg.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    p.mul_(residual_scale)

    def _init_weights(self, module):
        # Standard GPT-2/LLaMA-style init: small normal on projections and
        # embeddings. The residual-stream-writing projections get an
        # ADDITIONAL depth-dependent rescale afterward -- see __init__'s own
        # second pass, right after this gets called via self.apply.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, input_ids, labels=None):
        """input_ids: (B, T) long. labels: (B, T) long or None -- if given,
        returns (logits, loss) with the standard shifted next-token
        cross-entropy (loss is None if labels is None). Callers that
        already have shifted (x, y) pairs (see pretraining.shard_dataset)
        pass input_ids=x, labels=y directly -- no internal shifting here."""
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Minimal, unoptimized (no KV cache -- recomputes the full prefix
        every step) greedy/sampled generation, for qualitative sanity
        checks (see pretraining's own smoke test), not throughput. input_ids:
        (1, T) long. Returns (1, T + max_new_tokens)."""
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
