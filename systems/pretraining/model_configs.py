"""Named model-size presets, tiny through 7B, so `--model-size` picks a
sensible shape without hand-specifying every dimension. Every preset is a
LLaMA-style decoder-only transformer (see model.py); presets differ only in
scale, not architecture family.

vocab_size is deliberately not part of a preset -- it comes from whichever
systems/ tokenizer checkpoint is in use (TokenizerAdapter.vocab_size), so
the same --model-size preset yields a differently-sized embedding table
depending on the tokenizer, keeping scale and vocabulary independent.
"""

import dataclasses


def _swiglu_intermediate_size(hidden_size, multiple_of=256):
    """LLaMA convention: a SwiGLU MLP's natural width is 8/3 * hidden_size
    (vs. plain GELU's 4x -- the extra gating projection costs a third
    matmul, so per-matmul width shrinks to keep FLOPs comparable), rounded
    up to a multiple of `multiple_of` for GPU-friendly shapes."""
    raw = int(8 * hidden_size / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


@dataclasses.dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 0  # 0 means == num_heads (no GQA); lower enables
    # grouped-query attention (used by the 7b preset) to cut KV-cache size.
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0  # standard RoPE base frequency (LLaMA default)
    norm_eps: float = 1e-5
    dropout: float = 0.0  # standard for large-scale pretraining (vs. nonzero
    # dropout common in small-data finetuning)
    tie_embeddings: bool = True  # share input embedding/output projection
    # (GPT-2/LLaMA-small convention) -- saves vocab_size*hidden_size params;
    # presets below untie this for the largest tiers, matching LLaMA.
    grad_checkpointing: bool = False  # recompute activations in the backward
    # pass instead of storing them -- essential at larger presets to fit
    # GPU memory, pure overhead at tiny/small scale.

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
# lm_head add ~vocab_size * hidden_size * (1 or 2) on top -- negligible at
# 7B, not negligible at "tiny", so tiny's count is only a ballpark).
PRESETS = {
    "tiny": _preset(128, 4, 4),  # ~2-3M body params -- smoke testing only
    "small": _preset(768, 12, 12),  # ~85M body params -- GPT-2-small scale
    "medium": _preset(1024, 24, 16),  # ~300M body params
    "large": _preset(1536, 24, 16),  # ~700M body params. grad_checkpointing
    # deliberately OFF here (unlike xl/7b below): confirmed live on a real
    # 4xA100 run that "large"'s weights+optimizer+activations use only
    # ~27.7GB/80GB (34%) per GPU even without it -- checkpointing's whole
    # point (trading compute for memory) buys nothing at this size, while
    # its real cost (recomputing the forward pass during backward, ~6N ->
    # ~8N FLOPs/token) was actively capping this preset's own achieved MFU
    # (~25-34% of A100 peak on that same run, despite GPUs reading 100%
    # "busy" the whole time -- utilization != efficiency, see train.py's
    # own tokens_per_param/estimated_flops logging for how to recompute this
    # for a real run).
    "xl": _preset(
        2560, 32, 32, tie_embeddings=False, grad_checkpointing=True
    ),  # ~2.7B body params
    "7b": _preset(
        4096,
        32,
        32,
        num_kv_heads=8,  # GQA, 4 query heads/KV head -- matches later LLaMA/
        # Mistral practice (not LLaMA-1-7B's plain MHA); shrinks KV cache at
        # negligible quality cost.
        max_seq_len=4096,
        tie_embeddings=False,
        grad_checkpointing=True,
    ),  # ~5.9B params total -- matches LLaMA-7B's hidden_size/num_layers/
    # num_heads/intermediate_size (4096/32/32/11008) but lands below its
    # ~6.7B because of the GQA choice above (smaller k_proj/v_proj), which
    # is the intended effect, not a discrepancy. Needs multiple GPUs'
    # combined memory -- plain DDP (which replicates the full model per
    # rank) can't provide that, but train.py's TrainConfig.sharding="fsdp"
    # now can (see train.py's own module docstring); hasn't been run
    # end-to-end yet either way, unlike every smaller preset.
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown model size {name!r} -- choose from {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])  # copy so callers mutating
    # their instance (e.g. overriding max_seq_len) don't mutate the shared preset
