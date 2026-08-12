"""Named model-size presets, tiny through 7B, so `--model-size` picks a
sensible shape without hand-specifying every dimension. Every preset is a
LLaMA-style decoder-only transformer (see model.py) -- what differs is
purely scale (hidden_size/num_layers/num_heads/intermediate_size), not
architecture family.

vocab_size is deliberately NOT part of a preset: it comes from whichever
systems/ tokenizer checkpoint is actually being used (see
pretraining.tokenizer_adapter.TokenizerAdapter.vocab_size), so the same
--model-size preset produces a differently-sized embedding table depending
on which tokenizer feeds it -- exactly the point of keeping the two
concerns (architecture scale vs. vocabulary) independent.
"""

import dataclasses


def _swiglu_intermediate_size(hidden_size, multiple_of=256):
    """LLaMA's own convention (see the original LLaMA paper/code): a SwiGLU
    MLP's natural width is 8/3 * hidden_size (vs. a plain GELU MLP's 4x --
    SwiGLU's extra gating projection costs a third matmul, so the per-matmul
    width shrinks to keep total FLOPs comparable), then rounded UP to the
    nearest multiple of `multiple_of` for GPU-friendly tensor shapes."""
    raw = int(8 * hidden_size / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


@dataclasses.dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 0  # 0 means "== num_heads" (no grouped-query
    # attention); set lower than num_heads for GQA, which the 7b preset uses
    # (see PRESETS below) -- standard practice at that scale to cut KV-cache
    # size, matching e.g. LLaMA 2 70B/Mistral's own use of GQA.
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0  # standard RoPE base frequency -- LLaMA's own
    # default; only really matters to retune for context lengths well beyond
    # what's trained here.
    norm_eps: float = 1e-5
    dropout: float = 0.0  # 0.0 is standard for large-scale LM pretraining
    # (regularization matters much less than for small-data finetuning,
    # where dropout is more commonly nonzero).
    tie_embeddings: bool = True  # share the input embedding and output
    # projection weight matrix (GPT-2/LLaMA-small convention) -- saves
    # vocab_size * hidden_size parameters, meaningful at small-vocab/
    # small-hidden scale, negligible at 7B. Presets below override this
    # for the largest tiers, where LLaMA itself uses untied weights.
    grad_checkpointing: bool = False  # recompute activations during the
    # backward pass instead of storing them -- trades compute for memory;
    # ESSENTIAL at the larger presets to fit on real GPU memory, pure
    # overhead at tiny/small scale where memory was never the constraint.

    def num_kv_heads_resolved(self):
        return self.num_kv_heads or self.num_heads


def _preset(hidden_size, num_layers, num_heads, num_kv_heads=0, **overrides):
    return ModelConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=_swiglu_intermediate_size(hidden_size),
        **overrides,
    )


# Rough parameter counts below are the transformer body only (embedding +
# lm_head add roughly vocab_size * hidden_size * (1 or 2, depending on
# tie_embeddings) on top -- small relative to the body at 7B, NOT small at
# "tiny", which is why tiny's own count is only a ballpark).
PRESETS = {
    "tiny": _preset(128, 4, 4),  # ~2-3M body params -- fast local smoke
    # testing (see pretraining's own end-to-end test), not a real
    # pretraining target.
    "small": _preset(768, 12, 12),  # ~85M body params -- GPT-2-small scale.
    "medium": _preset(1024, 24, 16),  # ~300M body params.
    "large": _preset(1536, 24, 16, grad_checkpointing=True),  # ~700M body
    # params -- grad checkpointing on by default from here up.
    "xl": _preset(
        2560, 32, 32, tie_embeddings=False, grad_checkpointing=True
    ),  # ~2.7B body params.
    "7b": _preset(
        4096,
        32,
        32,
        num_kv_heads=8,  # grouped-query attention, 4 query heads per KV head --
        # matches later LLaMA/Mistral-family practice at this scale, not the
        # original LLaMA-1-7B (which used plain MHA) -- a deliberate choice
        # here since GQA meaningfully shrinks the KV cache at 7B with
        # negligible quality cost, and this project has no reason to
        # reproduce LLaMA-1 exactly rather than current best practice.
        max_seq_len=4096,
        tie_embeddings=False,
        grad_checkpointing=True,
    ),  # ~5.9B params total, confirmed by direct instantiation (torch's
    # `meta` device, no actual memory allocated) -- matches LLaMA-7B's
    # hidden_size/num_layers/num_heads/intermediate_size exactly
    # (4096/32/32/11008), but LANDS BELOW LLaMA-7B's own ~6.7B specifically
    # BECAUSE of the GQA choice above (num_kv_heads=8 shrinks k_proj/v_proj
    # relative to LLaMA-1's plain 32-head MHA) -- the parameter reduction is
    # the intended effect of GQA, not a discrepancy to explain away. This
    # WILL need multiple GPUs' combined memory (no FSDP/sharding yet -- see
    # train.py's module docstring) and has not been run end-to-end on this
    # project's own hardware, unlike every smaller preset.
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown model size {name!r} -- choose from {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])  # copy, so callers mutating
    # their own instance (e.g. overriding max_seq_len) never mutates the
    # shared preset.
