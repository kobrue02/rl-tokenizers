#!/bin/bash
#SBATCH --job-name=pretrain_train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Pretraining -- see systems/pretraining/train.py's own module docstring for what's
# actually verified: single-GPU and multi-GPU DistributedDataParallel (this
# script launches via torchrun automatically whenever more than one GPU was
# allocated -- see SLURM_GPUS_ON_NODE handling below). NOT yet built: FSDP/
# model sharding -- the --model-size 7b preset needs more memory than plain
# DDP's full-model-per-GPU replication can provide across any number of
# A100s, so a real 7B run is out of reach until that follow-up work exists.
# Every smaller preset (tiny/small/medium/large/xl) is fine under plain DDP.
#
# --gres=gpu:1 above is a single-GPU DEFAULT, not a hard limit: edit it to
# e.g. --gres=gpu:4 for a multi-GPU run on one node -- this script detects
# $SLURM_GPUS_ON_NODE and switches to torchrun automatically, no other
# change needed. Multi-NODE (several machines' GPUs in one job) is not
# handled here -- torchrun's own --nnodes/--rdzv-* flags would be the next
# step if a single node's GPUs stop being enough (still short of true FSDP
# sharding, just more DDP replicas).
#
# AUTO-RESUBMIT: a run whose token/step budget exceeds this job's own
# --time limit (e.g. the "large" preset's ~10-day estimate against this
# job's 24h default) will get SIGTERM/SIGKILL'd by SLURM mid-loop, well
# before train.py's own final.pt save at the bottom of its training loop
# (systems/pretraining/train.py) ever runs. Rather than requiring a human to notice
# the timeout and manually resubmit with --resume-from every time, this
# script checks for itself once training exits:
#   - final.pt present in output_dir -> genuinely done, exit 0, no resubmit.
#   - no final.pt, but a NEWER step_*.pt checkpoint exists than when this
#     run started -> real progress was made this run (just not enough to
#     finish); resubmit itself via sbatch with --resume-from that
#     checkpoint, preserving this run's own --gres GPU count (which a bare
#     `sbatch jobs/train_pretraining.sh ...` would NOT do on its own --
#     it would silently fall back to this script's #SBATCH --gres=gpu:1
#     default, quietly dropping a multi-GPU run back to 1 GPU on every
#     resume).
#   - no final.pt AND no checkpoint progress beyond this run's own starting
#     point -> treated as a genuine failure (e.g. a persistent crash before
#     the first save_steps checkpoint), NOT resubmitted, so a real bug
#     can't spin into an infinite resubmission loop burning allocation.
# This means squeue will show a NEW job id appear every ~24h for a
# multi-day run until it finishes -- expected, not a bug -- and
# --mail-type=ALL will send one completion/failure email per segment.
#
# Usage:
#   sbatch jobs/train_pretraining.sh --shard-dir pretrain_data/glot500_bpe \
#       --model-size small --total-steps 50000 --seq-len 1024 --per-device-batch-size 16
#   sbatch --gres=gpu:4 jobs/train_pretraining.sh -c configs/pretrain_fanta_large.yml
#
# All flags forward directly to systems/pretraining/cli.py -- see
# `python3 -m systems.pretraining.cli --help`.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # see module docstring above -- avoids the module's
# own (older) cuDNN shadowing PyTorch's bundled one.

# 3. Environment
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_EXTENSIONS_DIR"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints/pretrain

# 5. Resolve this run's output_dir/total_steps from the EXACT args this job
# received (config file + any CLI overrides, e.g. a --resume-from appended
# by a prior resubmit of this same script) -- reuses systems.pretraining.cli's own
# parsing so this can never drift from what systems.pretraining.cli itself uses.
CFG_INFO=$(python3 -c "
import sys
from systems.pretraining.cli import build_arg_parser, _config_from_args
from common.config_file import parse_args_with_config
args = parse_args_with_config(build_arg_parser(), sys.argv[1:])
cfg = _config_from_args(args)
print(cfg.output_dir)
print(cfg.total_steps)
" "$@")
OUTPUT_DIR=$(echo "$CFG_INFO" | sed -n '1p')
TOTAL_STEPS=$(echo "$CFG_INFO" | sed -n '2p')
echo "Resolved output_dir=$OUTPUT_DIR total_steps=$TOTAL_STEPS"

# Highest-step step_*.pt in a dir, empty string if none -- used both before
# and after the run below purely to detect whether THIS run made progress,
# not as the resume mechanism itself (train.py's own --resume-from/
# load_checkpoint handles that).
latest_checkpoint() {
    ls "$1"/step_*.pt 2>/dev/null | sed -E 's#.*/step_([0-9]+)\.pt#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-
}

BEFORE_CKPT=$(latest_checkpoint "$OUTPUT_DIR")
if [ -n "$BEFORE_CKPT" ]; then
    BEFORE_STEP=$(basename "$BEFORE_CKPT" | sed -E 's/step_([0-9]+)\.pt/\1/')
else
    BEFORE_STEP=0
fi

# 6. Run -- single process for one GPU, torchrun for more than one.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting pretraining with $NUM_GPUS GPU(s), args: $@"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m systems.pretraining.cli "$@"
else
    python3 -m systems.pretraining.cli "$@"
fi
TRAIN_EXIT=$?

# 7. Done, or resubmit? See the AUTO-RESUBMIT comment at the top.
if [ -f "$OUTPUT_DIR/final.pt" ]; then
    echo "Training complete -- reached total_steps=$TOTAL_STEPS, final.pt written."
    exit 0
fi

echo "final.pt not found in $OUTPUT_DIR (training exited with code $TRAIN_EXIT) -- checking for progress to resume from."
AFTER_CKPT=$(latest_checkpoint "$OUTPUT_DIR")
if [ -z "$AFTER_CKPT" ]; then
    echo "No checkpoint found in $OUTPUT_DIR at all -- treating this as a real failure, NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi
AFTER_STEP=$(basename "$AFTER_CKPT" | sed -E 's/step_([0-9]+)\.pt/\1/')
if [ "$AFTER_STEP" -le "$BEFORE_STEP" ]; then
    echo "Latest checkpoint step ($AFTER_STEP) did not advance past this run's own starting point ($BEFORE_STEP) -- no real progress was made, NOT resubmitting (likely a persistent crash). Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi

# This job's OWN actual --time limit (may have been overridden at
# submission) -- a bare resubmit would otherwise silently fall back to this
# script's #SBATCH --time=24:00:00 default, same reasoning as --gres above.
TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

echo "Progress made this run: step $BEFORE_STEP -> $AFTER_STEP (of $TOTAL_STEPS). Resubmitting from $AFTER_CKPT..."
sbatch --gres=gpu:"$NUM_GPUS" --time="$TIME_LIMIT" jobs/train_pretraining.sh "$@" --resume-from "$AFTER_CKPT"
SBATCH_EXIT=$?
if [ "$SBATCH_EXIT" -ne 0 ]; then
    echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
    echo "  sbatch --gres=gpu:$NUM_GPUS --time=$TIME_LIMIT jobs/train_pretraining.sh $@ --resume-from $AFTER_CKPT" >&2
    exit 1
fi
echo "Resubmitted successfully."
exit 0
