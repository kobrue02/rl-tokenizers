"""A LLaMA-style decoder-only transformer: RMSNorm, rotary position
embeddings (RoPE), SwiGLU-gated MLP, causal self-attention (optionally
grouped-query, for the largest presets) via torch's fused
scaled_dot_product_attention -- the architecture essentially every current
open LLM (LLaMA, Mistral, Qwen, Gemma, ...) uses. See model_configs.py for
the named size presets built on this one architecture.

Unlike the tokenizer baselines in systems/, this isn't a scaled-down
simplification -- it uses real architectural choices, at whatever
--model-size the caller picks (see train.py and model_configs.PRESETS).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: normalizes by RMS only (no mean
    subtraction, no bias) -- cheaper than LayerNorm, LLaMA's norm of choice."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Computed in fp32 regardless of ambient autocast dtype -- the
        # reciprocal-sqrt reduction loses meaningful precision in bf16
        # otherwise (a known LLaMA implementation detail).
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(dtype)


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):
    """Returns (cos, sin), each (max_seq_len, head_dim) -- head_dim (not
    head_dim/2) because each half of a rotation pair repeats the same
    cos/sin value (GPT-NeoX/LLaMA style, not the original RoFormer
    interleaved-pairs style; the two are equivalent up to a fixed
    permutation of dimensions)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )  # (head_dim/2,)
    t = torch.arange(max_seq_len, device=device).float()  # (T,)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return freqs.cos(), freqs.sin()


def _rotate_half(x):
    """x: (..., head_dim). Splits in half and swaps-with-negation -- the
    GPT-NeoX-style rotation companion to precompute_rope's cos/sin layout."""
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
    F.scaled_dot_product_attention rather than hand-rolled softmax(QK^T/
    sqrt(d))V so PyTorch can dispatch to a fused/flash kernel -- a
    hand-rolled version materializes a full (T,T) score matrix per head,
    the same O(T^2) cost that OOM'd systems.tokenization.fanta's dense-attention
    baseline at far smaller scale."""

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
            f"{self.rope_cos.shape[0]} -- RoPE has no precomputed frequencies past that"
        )
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:T].to(dtype=q.dtype, device=q.device)
        sin = self.rope_sin[:T].to(dtype=q.dtype, device=q.device)
        q, k = apply_rope(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            # GQA: each KV head is shared by num_heads/num_kv_heads query
            # heads; repeat_interleave so query head i uses kv head i//group_size.
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


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Cross-entropy of hidden @ lm_head_weight.T against labels, computed
    in row-chunks so the (chunk_size, vocab_size) logits slab is the only
    one ever materialized -- never the full (num_rows, vocab_size) tensor a
    plain `lm_head(x)` + F.cross_entropy would keep alive for backward.
    Trades that memory for one extra lm_head-sized matmul: backward
    recomputes each chunk's logits from the saved (hidden, weight) rather
    than reusing logits saved from forward, since forward never kept them
    around in the first place."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size, ignore_index):
        num_rows = hidden.shape[0]
        total_loss = hidden.new_zeros(())
        total_count = 0
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk_labels = labels[start:end]
            n_valid = int((chunk_labels != ignore_index).sum())
            if n_valid == 0:
                continue
            chunk_logits = hidden[start:end] @ weight.t()
            total_loss = total_loss + F.cross_entropy(
                chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
            )
            total_count += n_valid
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.total_count = total_count  # 0 possible (a batch entirely of
        # ignore_index) -- guarded in both forward's division and backward's
        # scaling below rather than letting either divide by zero.
        return total_loss / max(total_count, 1)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels = ctx.saved_tensors
        if ctx.total_count == 0:
            return torch.zeros_like(hidden), torch.zeros_like(weight), None, None, None
        chunk_size, ignore_index = ctx.chunk_size, ctx.ignore_index
        scale = grad_output / ctx.total_count
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        # The autograd engine runs Function.backward with grad mode globally
        # OFF (unless the caller requested create_graph=True, which nothing
        # here does) -- the recomputation below needs it explicitly back on,
        # or chunk_logits/chunk_loss build no graph at all and
        # torch.autograd.grad has nothing to differentiate through (same
        # requirement torch.utils.checkpoint's own backward has, for the
        # same reason).
        with torch.enable_grad():
            for start in range(0, hidden.shape[0], chunk_size):
                end = min(start + chunk_size, hidden.shape[0])
                chunk_labels = labels[start:end]
                if not (chunk_labels != ignore_index).any():
                    continue
                # Fresh leaves, detached from `hidden`/`weight`'s own
                # history -- this recomputation is what lets forward skip
                # saving logits.
                chunk_hidden = hidden[start:end].detach().requires_grad_(True)
                chunk_weight = weight.detach().requires_grad_(True)
                chunk_logits = chunk_hidden @ chunk_weight.t()
                chunk_loss = F.cross_entropy(
                    chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
                )
                g_hidden, g_weight = torch.autograd.grad(chunk_loss * scale, [chunk_hidden, chunk_weight])
                grad_hidden[start:end] = g_hidden
                grad_weight += g_weight
        return grad_hidden, grad_weight, None, None, None


def chunked_cross_entropy(hidden, weight, labels, chunk_size, ignore_index=-100):
    """hidden: (num_rows, hidden_size) -- already flattened over batch*seq.
    weight: lm_head.weight, (vocab_size, hidden_size). labels: (num_rows,).
    Returns a scalar loss, numerically and gradient-wise equivalent to
    F.cross_entropy((hidden @ weight.T), labels, ignore_index=ignore_index)
    up to float rounding from chunking's different summation order -- see
    _ChunkedCrossEntropy for why this trades one extra matmul for never
    materializing the full logits tensor."""
    return _ChunkedCrossEntropy.apply(hidden, weight, labels, chunk_size, ignore_index)


def _padded_vocab_size(vocab_size, multiple_of=64):
    """Rounds vocab_size up to a multiple of `multiple_of` for the
    embedding/lm_head matmul shapes -- a real tensor-core throughput win
    (matches lit-llama's identical find_multiple(vocab_size, 64)
    convention), not cosmetic: none of this project's own tokenizer vocab
    sizes are naturally multiples of 64. The extra rows/columns are never
    indexed by a real token id (ids are always < the TRUE vocab_size,
    tracked separately as TransformerLM.vocab_size) and never appear in a
    training label, so the loss naturally drives their logits toward
    -inf over training; TransformerLM.generate() explicitly masks them out
    before sampling so an under-trained model can never sample one."""
    if vocab_size % multiple_of == 0:
        return vocab_size
    return vocab_size + multiple_of - (vocab_size % multiple_of)


class TransformerLM(nn.Module):
    """Full decoder-only LM: token embedding -> N TransformerBlocks -> final
    RMSNorm -> output projection to vocab logits. vocab_size is passed
    separately from `cfg` (purely architectural) since it comes from
    whichever tokenizer is in use, not the chosen model size."""

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size  # the TRUE vocab size -- see _padded_vocab_size
        padded_vocab_size = _padded_vocab_size(vocab_size)
        self.embed = nn.Embedding(padded_vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, padded_vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        # Second, targeted pass: rescale the two per-block projections that
        # write into the residual stream (attn.o_proj, mlp.down_proj) by
        # 1/sqrt(2*num_layers) -- GPT-2's fix for residual variance growing
        # with depth. Needs dotted parameter names to target just these two
        # Linear layers, which _init_weights (called per-module via
        # nn.Module.apply) can't see -- hence a separate named_parameters() pass.
        residual_scale = 1.0 / math.sqrt(2 * cfg.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    p.mul_(residual_scale)

    def _init_weights(self, module):
        # Standard GPT-2/LLaMA-style init: small normal on projections and
        # embeddings. Residual-stream-writing projections get an additional
        # depth-dependent rescale afterward (see __init__'s second pass).
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
        returns (logits, loss) with standard next-token cross-entropy,
        UNLESS cfg.loss_chunk_size > 0, in which case logits is None (see
        chunked_cross_entropy's docstring for why: the whole point is to
        never materialize the full (B*T, padded_vocab_size) tensor). Every
        current caller that passes labels only ever reads back the loss
        (train.py, validate()) -- confirmed via a repo-wide grep before
        making this trade. Callers with already-shifted (x, y) pairs
        (shard_dataset) pass input_ids=x, labels=y directly; no internal
        shifting here."""
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)

        if labels is not None and self.cfg.loss_chunk_size > 0:
            loss = chunked_cross_entropy(
                x.reshape(-1, x.size(-1)),
                self.lm_head.weight,
                labels.reshape(-1),
                self.cfg.loss_chunk_size,
            )
            return None, loss

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
            # self(context), NOT self.forward(context): the latter calls
            # forward() directly, bypassing nn.Module.__call__ entirely --
            # harmless for every CURRENT caller (a plain, unwrapped model
            # has no hooks registered either way), but FSDP2's own
            # unshard/reshard mechanism for this model's TOP-LEVEL
            # parameters (embed/norm/lm_head, see train.wrap_fsdp) is
            # implemented as exactly such a hook. Calling generate() on an
            # FSDP-wrapped model via self.forward() would silently run
            # with unsharded/incomplete top-level parameters -- confirmed
            # this is the only broken piece: forward()'s own submodule
            # calls (block(x), not block.forward(x)) already go through
            # __call__ correctly, so per-block FSDP hooks were never the
            # problem.
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if next_logits.size(-1) > self.vocab_size:
                # Padding slots (see _padded_vocab_size) never appear in
                # training labels, but an under-trained model could still
                # assign them nonzero probability -- mask them out so
                # sampling can never return an invalid (>= true vocab_size) id.
                next_logits[:, self.vocab_size :] = float("-inf")
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
