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

# Pretraining (systems/pretraining/train.py) -- single-GPU or multi-GPU DDP
# (torchrun, auto-detected via SLURM_GPUS_ON_NODE). No FSDP yet, so --model-size
# 7b isn't runnable; every smaller preset (tiny/small/medium/large/xl) is fine.
# --gres=gpu:1 is a single-GPU DEFAULT -- override with e.g. --gres=gpu:4 for
# multi-GPU on one node (multi-NODE isn't handled here). --cpus-per-task=16
# is also a DEFAULT sized for gpu:1 -- ALSO override it at higher GPU counts
# (e.g. --cpus-per-task=32 alongside --gres=gpu:8), since this cluster ties
# memory to cpus-per-task at ~1.95GB/core (see jobs/train_parity_bpe.sh's own
# comment) and each rank's DataLoader spawns TrainConfig.num_workers=4 worker
# processes on top of its own main process. Confirm --nodes=1 --gres=gpu:8 is
# actually schedulable on gpu_a100_il before relying on it (node GPU count
# isn't recorded anywhere in this repo) -- e.g. `sinfo -p gpu_a100_il -o "%n %G"`.
#
# AUTO-RESUBMIT: a run whose budget exceeds this job's --time limit gets
# killed mid-loop before final.pt is written. This script checks after exit:
# final.pt present -> done, exit 0. No final.pt but a newer step_*.pt than
# when this run started -> real progress, resubmit with --resume-from
# (preserving this run's own --gres/--time/--cpus-per-task, which a bare
# resubmit wouldn't).
# No progress at all -> real failure, NOT resubmitted. Expect a new job id
# in squeue roughly every 24h for a multi-day run -- that's normal.
#
# Usage:
#   sbatch jobs/train_pretraining.sh --shard-dir pretrain_data/glot500_bpe \
#       --model-size small --total-steps 50000 --seq-len 1024 --per-device-batch-size 16
#   sbatch --gres=gpu:4 jobs/train_pretraining.sh -c configs/pretrain_fanta_large.yml
#   sbatch --gres=gpu:8 --cpus-per-task=32 jobs/train_pretraining.sh -c configs/pretrain_bpe_large.yml

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # avoids the cuda module's older cuDNN shadowing PyTorch's bundled one

export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_EXTENSIONS_DIR"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints/pretrain

# Resolve output_dir/total_steps from the exact args this job received --
# reuses systems.pretraining.cli's own parsing so this can't drift from it.
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

latest_checkpoint() {
    ls "$1"/step_*.pt 2>/dev/null | sed -E 's#.*/step_([0-9]+)\.pt#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-
}

BEFORE_CKPT=$(latest_checkpoint "$OUTPUT_DIR")
if [ -n "$BEFORE_CKPT" ]; then
    BEFORE_STEP=$(basename "$BEFORE_CKPT" | sed -E 's/step_([0-9]+)\.pt/\1/')
else
    BEFORE_STEP=0
fi

# Single process for one GPU, torchrun for more than one.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting pretraining with $NUM_GPUS GPU(s), args: $@"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m systems.pretraining.cli "$@"
else
    python3 -m systems.pretraining.cli "$@"
fi
TRAIN_EXIT=$?

# Done, or resubmit? See AUTO-RESUBMIT above.
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

TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')
# Preserve THIS job's own --cpus-per-task too, not just --gres/--time -- this
# cluster ties memory to cpus-per-task at ~1.95GB/core (see
# jobs/train_parity_bpe.sh's own comment), and a multi-GPU run needs more
# than the script's #SBATCH --cpus-per-task=16 default (8 GPUs x
# TrainConfig.num_workers=4 DataLoader workers = 32 processes alone). Without
# this, a run launched with an explicit --cpus-per-task override loses it on
# every resubmit after the first, silently reverting to 16 and risking an
# OOM on the next checkpoint load.
CPUS_PER_TASK=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'CPUs/Task=\K\S+')

echo "Progress made this run: step $BEFORE_STEP -> $AFTER_STEP (of $TOTAL_STEPS). Resubmitting from $AFTER_CKPT..."
sbatch --gres=gpu:"$NUM_GPUS" --time="$TIME_LIMIT" --cpus-per-task="$CPUS_PER_TASK" jobs/train_pretraining.sh "$@" --resume-from "$AFTER_CKPT"
SBATCH_EXIT=$?
if [ "$SBATCH_EXIT" -ne 0 ]; then
    echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
    echo "  sbatch --gres=gpu:$NUM_GPUS --time=$TIME_LIMIT --cpus-per-task=$CPUS_PER_TASK jobs/train_pretraining.sh $@ --resume-from $AFTER_CKPT" >&2
    exit 1
fi
echo "Resubmitted successfully."
exit 0
