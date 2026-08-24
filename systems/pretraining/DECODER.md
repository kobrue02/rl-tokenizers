# Decoder pipeline (causal LM, LLaMA architecture)

A from-scratch LLaMA-style decoder-only transformer (`model.py`: RMSNorm,
RoPE, SwiGLU, causal attention via `F.scaled_dot_product_attention`,
grouped-query attention on the largest preset) — six named size presets in
`model_configs.py`, `tiny` through `7b`:

| preset | hidden | layers | heads | GQA kv-heads |
|---|---|---|---|---|
| `tiny` | 128 | 4 | 4 | — (smoke testing only) |
| `small` | 768 | 12 | 12 | — |
| `medium` | 1024 | 24 | 16 | — |
| `large` | 1536 | 24 | 16 | — |
| `xl` | 2560 | 32 | 32 | — |
| `7b` | 4096 | 32 | 32 | 8 |

Every CLI below accepts `-c`/`--config path/to/file.yml` — see
`configs/README.md` for the full precedence rules and worked examples of a
complete bpe/fanta experiment end to end. Run
`python3 -m systems.pretraining.<module> --help` for the complete flag
list of any command; this file covers the shape of the pipeline. See
**`EVALS.md`** for evaluating a trained checkpoint (benchmark suite +
contamination checking) and **`ENCODER.md`** for the separate MLM encoder
pipeline that shares this same tokenizer/data-prep stage.

## 1. Data prep: corpus → packed token shards

```bash
sbatch jobs/prep_pretraining_data.sh --dataset glot500 --langs all \
    --dataset-config "$WORK_ROOT/data/glot500" \
    --system bpe --checkpoint checkpoints/bpe_50k.json \
    --output-dir pretrain_data/glot500_bpe --max-tokens 5000000000
```

- **`--dataset glot500`/`bible_nlp`** need a one-time local disk cache
  first (`sbatch jobs/prepare_glot500.sh` / `prepare_bible_nlp.sh`) — no
  live-streaming fallback, since re-streaming a ~300GB/~411-config corpus
  on every resume was a real measured bottleneck.
- **CPU vs GPU job**: `prep_pretraining_data.sh` (CPU) is fine for
  bpe/superbpe. The five neural/span-family systems (`fanta`, `magnet`,
  `manta`, `flexitokens`, `fairtok`) need `prep_pretraining_data_gpu.sh`
  instead — their `induce_spans` is a real forward pass per document, far
  slower on CPU — and need `--vocab-json` too (bpe/superbpe don't).
- **Resumable**: rerunning the same command against the same
  `--output-dir` after a mid-run kill (e.g. a SLURM time limit) resumes
  automatically from `prep_checkpoint.json` — no extra flag needed.
- `--dedup`/`--no-dedup` (on by default), `--prefetch` (off by default —
  overlaps network-bound streaming with dedup/encode/pack work, but
  verify with a real timing comparison before relying on it for a
  full-scale run) are the two performance knobs most worth knowing about;
  see `--help` for the rest (bucketing, truncation, dedup thresholds).

## 2. Pretraining

```bash
sbatch jobs/train_pretraining.sh --shard-dir pretrain_data/glot500_bpe \
    --model-size small --total-steps 50000 --seq-len 1024 --per-device-batch-size 16
```

- **Give every run its own `--output-dir`** — defaults to
  `checkpoints/pretrain` regardless of `--shard-dir`; two concurrent runs
  at the default will overwrite each other's checkpoints.
- **`--sharding ddp`** (default) or **`--sharding fsdp`** — FSDP2 is what
  actually makes the `7b` preset runnable at all (DDP replicates the full
  model per rank; `7b`'s bf16 weights+gradients+AdamW state alone need
  ~73GB, too much for one 80GB card). Every smaller preset works fine
  under plain DDP.
- **Multi-GPU**: `--gres=gpu:N` (`torchrun` is auto-invoked once
  `SLURM_GPUS_ON_NODE > 1`); scale `--cpus-per-task` with it (this
  cluster ties memory to cpus-per-task, and each rank spawns
  `--num-workers` DataLoader workers on top of its own main process).
- **`--compile`** (torch.compile) and **`--loss-chunk-size`** (chunked
  cross-entropy, trading one extra matmul for never materializing the
  full `(batch*seq_len, vocab)` logits tensor) are both off by default —
  real perf wins, but unverified-by-default on real hardware; measure
  before relying on either for a full run.
- **Auto-resubmit**: a run whose budget exceeds this job's `--time` limit
  gets killed mid-loop; the script detects real progress (a newer
  `step_*.pt` than when it started) and resubmits itself with
  `--resume-from`, preserving `--gres`/`--time`/`--cpus-per-task`. No
  progress at all → treated as a real failure, not resubmitted.

## 3. Qualitative generation

`cli_generate.py` has three usage modes — the only CLI in this pipeline
with an interactive one:

```bash
# Interactive "app" mode: browse checkpoints on disk, pick one, type prompts in a loop
python3 -m systems.pretraining.cli_generate

# Interactive, pointed at a specific checkpoint (skips the browse step)
python3 -m systems.pretraining.cli_generate --checkpoint checkpoints/pretrain/final.pt

# Batch/scripted (as part of a SLURM pipeline) -- --prompt is repeatable
sbatch --dependency=afterok:$train_id jobs/generate_samples.sh \
    --checkpoint checkpoints/pretrain/final.pt \
    --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
    --prompt "The quick brown fox" --prompt "Once upon a time" \
    --max-new-tokens 100 --num-samples 3 --output results/samples_bpe.json
```

In interactive mode, `--system`/`--tokenizer-checkpoint` are auto-resolved
from the checkpoint's own stored `shard_dir` → that directory's
`shards_meta.json` — only asked for explicitly (`--vocab-json`) if the
resolved tokenizer actually needs it. Passing `--system`/
`--tokenizer-checkpoint`/`--prompt` together is what selects batch mode
over either interactive one.

Generation has no KV cache (recomputes the full prefix every step) — fine
for qualitative spot-checks, not meant for throughput; `TransformerLM.generate`'s
own docstring covers why.

## 4. wandb logging

All four decoder CLIs (`data_prep`, `cli` (training), `cli_generate`,
`cli_eval`) share the `pretraining` wandb project by default — filter by
`job_type` (`data_prep`, `train`, `generate`, `eval`; `cli_contamination`
adds `contamination_check`) to tell stages apart in the UI.
