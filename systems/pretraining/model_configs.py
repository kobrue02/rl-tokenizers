"""Named model-size presets, tiny through 12B, so `--model-size` picks a
sensible shape without hand-specifying every dimension. Two architecture
families (see ModelConfig.architecture and model.py): "llama" (tiny through
7b below -- RMSNorm/SwiGLU/full rotary/sequential residual) and "gpt_neox"
(the "pythia_*" presets -- LayerNorm/GELU/partial rotary/parallel residual,
reproducing EleutherAI's Pythia suite). Presets within a family differ only
in scale, not architecture.

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


def _gelu_intermediate_size(hidden_size):
    """GPT-NeoX/Pythia convention: a plain (non-gated) GELU MLP is exactly
    4x hidden_size -- no rounding needed since there's no extra gating
    matmul narrowing it the way SwiGLU's is (see _swiglu_intermediate_size)."""
    return 4 * hidden_size


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
    architecture: str = "llama"  # "llama" (default -- RMSNorm, SwiGLU MLP,
    # full rotary embeddings, sequential pre-norm residual, GPT-2-style init
    # with a depth-rescaled residual write, exactly what every preset above
    # this field used before it existed) or "gpt_neox" (LayerNorm, plain
    # GELU MLP, partial rotary via rotary_pct below, PARALLEL residual --
    # x = x + attn(norm(x)) + mlp(norm(x)), one shared norm, not two
    # sequential ones -- and small_init/wang_init). See model.py's
    # TransformerBlock/TransformerLM for where this branches, and the
    # "pythia_*" presets below, which reproduce EleutherAI/gpt-neox's own
    # configs/pythia/*.yml (confirmed by fetching those files directly:
    # pos_emb=rotary+rotary_pct=0.25, gpt_j_residual=true, no_weight_tying=
    # true, norm defaults to layernorm, activation defaults to gelu,
    # init_method=small_init/output_layer_init_method=wang_init).
    rotary_pct: float = 1.0  # fraction of head_dim RoPE is applied to; the
    # remaining (1 - rotary_pct) fraction of each head passes through
    # unrotated. 1.0 (full rotary, LLaMA convention) unless overridden --
    # every "pythia_*" preset below sets 0.25, GPT-NeoX-20B/Pythia's own
    # choice (a full-precision ablation in the GPT-NeoX-20B paper found
    # partial rotary matches full rotary's quality at lower compute).
    grad_checkpointing: bool = False  # recompute activations in the backward
    # pass instead of storing them -- essential at larger presets to fit
    # GPU memory, pure overhead at tiny/small scale.
    loss_chunk_size: int = 0  # 0 disables (matches this project's other
    # 0-disables flags, e.g. eval_interval/generate_interval/max_doc_bytes).
    # >0 computes the LM loss in row-chunks of the lm_head projection (see
    # model.chunked_cross_entropy) instead of materializing the full
    # (batch*seq_len, padded_vocab_size) logits tensor at once -- the
    # dominant activation-memory cost at large vocab*batch*seq_len, unrelated
    # to grad_checkpointing (which covers the transformer blocks, not the
    # output projection). Off by default: real payoff needs measuring
    # tok/s and peak memory on real hardware first, same convention as
    # TrainConfig.compile/--prefetch elsewhere in this project.

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


def _gpt_neox_preset(hidden_size, num_layers, num_heads, **overrides):
    """Builds a "gpt_neox"-architecture ModelConfig at the given shape.
    grad_checkpointing intentionally does NOT blanket-copy every
    "pythia_*" preset's own checkpoint_activations=true (see
    configs/pythia/*.yml) -- this project's own "large" preset above found
    (via a real profiled run, see its own comment) that checkpointing can
    be a pure loss at a size that already fits comfortably in memory
    without it, so the same on-by-default-only-at-real-memory-pressure
    policy is kept here rather than importing GPT-NeoX's DeepSpeed-era
    default uncritically; overridable per preset below regardless."""
    return ModelConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=_gelu_intermediate_size(hidden_size),
        architecture="gpt_neox",
        rotary_pct=0.25,
        tie_embeddings=False,  # no_weight_tying: true in every pythia/*.yml
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
    # --- Pythia suite (EleutherAI/pythia's own README table, architecture
    # cross-checked directly against EleutherAI/gpt-neox's configs/pythia/
    # *.yml -- see ModelConfig.architecture's own docstring). hidden_size/
    # num_layers/num_heads below are exact; each preset's own comment gives
    # the paper's reported peak learning_rate (a TrainConfig field, not a
    # ModelConfig one) to set explicitly in that experiment's own
    # pretrain_*.yml -- e.g. `learning_rate: 6.0e-4` for pythia_160m --
    # plus train.py's own effective_batch_size (per_device_batch_size *
    # grad_accum_steps * world_size * seq_len) to reach the paper's 2M
    # tokens/step, and (matching Pythia's own cosine-to-1/10th-peak decay,
    # not this project's other presets' decay-to-zero) `min_lr_ratio: 0.1`
    # -- see TrainConfig.min_lr_ratio.
    "pythia_14m": _gpt_neox_preset(128, 6, 4),  # peak lr 1.0e-3
    "pythia_31m": _gpt_neox_preset(256, 6, 8),  # peak lr 1.0e-3
    "pythia_70m": _gpt_neox_preset(512, 6, 8),  # peak lr 1.0e-3
    "pythia_160m": _gpt_neox_preset(768, 12, 12),  # peak lr 6.0e-4
    "pythia_410m": _gpt_neox_preset(1024, 24, 16),  # peak lr 3.0e-4
    "pythia_1b": _gpt_neox_preset(2048, 16, 8),  # peak lr 3.0e-4
    "pythia_1_4b": _gpt_neox_preset(2048, 24, 16),  # peak lr 2.0e-4
    "pythia_2_8b": _gpt_neox_preset(
        2560, 32, 32, grad_checkpointing=True
    ),  # peak lr 1.6e-4
    "pythia_6_9b": _gpt_neox_preset(
        4096, 32, 32, grad_checkpointing=True
    ),  # peak lr 1.2e-4 -- needs FSDP, same as this project's own "7b" above
    "pythia_12b": _gpt_neox_preset(
        5120, 36, 40, grad_checkpointing=True
    ),  # peak lr 1.2e-4 -- needs FSDP, same as this project's own "7b" above
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown model size {name!r} -- choose from {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])  # copy so callers mutating
    # their instance (e.g. overriding max_seq_len) don't mutate the shared preset
