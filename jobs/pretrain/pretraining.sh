#!/bin/bash
#SBATCH --job-name=pretrain_train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
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
# memory to cpus-per-task at ~1.95GB/core (see jobs/train_tokenizer/parity_bpe.sh's own
# comment) and each rank's DataLoader spawns TrainConfig.num_workers=4 worker
# processes on top of its own main process. Confirm --nodes=1 --gres=gpu:8 is
# actually schedulable on gpu_a100_il before relying on it (node GPU count
# isn't recorded anywhere in this repo) -- e.g. `sinfo -p gpu_a100_il -o "%n %G"`.
#
# AUTO-RESUBMIT: a run whose budget exceeds this job's --time limit gets
# killed mid-loop before final.pt is written. This script checks: final.pt
# present -> done, exit 0. No final.pt but a newer step_*.pt than when this
# run started -> real progress, resubmit with --resume-from (preserving
# this run's own --gres/--time/--cpus-per-task/--partition, which a bare
# resubmit wouldn't). No progress at all -> real failure, NOT resubmitted.
#
# --signal=B:TERM@180 + the trap below is WHY this can actually fire on a
# real time-limit exit, not just an OOM/crash: a genuine SLURM TIMEOUT
# kills the ENTIRE job -- this wrapper script included, not just its
# python/torchrun child -- the instant the limit is reached, so code after
# a plain foreground command would never run at all (confirmed live on a
# real 12h data-prep TIMEOUT: its own log showed no "checking for
# progress"/"Resubmitted successfully" message whatsoever, unlike every
# OOM-kill case, which printed both -- the exact same risk applies here,
# just on a 24h cadence instead of 12h). --signal=B:TERM@180 asks SLURM to
# send SIGTERM to this SCRIPT 180s before the hard limit instead, caught
# by the trap below, which runs the exact same check-and-resubmit logic
# early, inside that 180s grace window, before SLURM's own kill lands.
# Expect a new job id in squeue roughly every 24h for a multi-day run --
# that's normal.
#
# LOCAL-SSD STAGING: ShardedTokenDataset re-reads random windows from
# shard_dir's own .bin files on every single training step for the entire
# run (hundreds of thousands of steps) -- exactly the access pattern
# bwUniCluster's own docs call out $TMPDIR (local node NVMe) for over the
# parallel Lustre filesystem ("data which is read many times on a single
# node... should be copied to $TMPDIR and read from there"), not just large
# sequential throughput Lustre is tuned for. This copies shard_dir to
# $TMPDIR once at job start and points --shard-dir at that local copy
# instead -- re-paid on every AUTO-RESUBMIT too, since $TMPDIR is purged
# between jobs (a fresh per-job directory, not a persistent cache), but at
# this project's actual shard sizes (a uint16-dtype "large"-preset corpus is
# tens of GB, not TBs) the one-time copy cost is trivial next to a
# multi-day run's own read volume.
#
# Usage:
#   sbatch jobs/pretrain/pretraining.sh --shard-dir pretrain_data/culturax_bpe_large \
#       --model-size small --total-steps 50000 --seq-len 1024 --per-device-batch-size 16
#   sbatch --partition=gpu_h100 --gres=gpu:4 jobs/pretrain/pretraining.sh -c configs/pretrain/bpe_culturax.yml
#   sbatch --partition=gpu_h100 --gres=gpu:4 jobs/pretrain/pretraining.sh -c configs/pretrain/fanta_culturax.yml
#   # gpu_h100 is bwUniCluster 3.0's DEDICATED H100 queue (AMD EPYC 9454,
#   # 94GiB/GPU, 15.36TB local NVMe) -- distinct from the shared
#   # gpu_a100_il/gpu_h100_il pool (Ice Lake, 80GiB/GPU, 6.4TB, either card
#   # type). Overriding --partition is preserved across AUTO-RESUBMIT the
#   # same way --gres/--time/--cpus-per-task already are (see below).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# Captured HERE, before anything else can consume/shift "$@" -- see
# jobs/prep/pretraining_data.sh's own comment for the full bug this fixes
# (check_and_resubmit() is always called bare -- from both the normal
# post-`wait` path and the on_term SIGTERM trap -- so a literal "$@" inside
# it is a function-local empty array, not this script's real arguments).
# Confirmed live: this is what silently dropped -c configs/pretrain/
# bpe_culturax.yml on a real 24h-boundary auto-resubmit, resolving
# output_dir back to the shared default "checkpoints/pretrain" instead of
# "checkpoints/pretrain_bpe_culturax_large" and crashing the run at 99.3%
# complete (step 206000/207383).
ORIG_ARGS=("$@")

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

# Resolve output_dir/total_steps/shard_dir from the exact args this job
# received -- reuses systems.pretraining.cli's own parsing so this can't
# drift from it.
CFG_INFO=$(python3 -c "
import sys
from systems.pretraining.cli import build_arg_parser, _config_from_args
from common.config_file import parse_args_with_config
args = parse_args_with_config(build_arg_parser(), sys.argv[1:])
cfg = _config_from_args(args)
print(cfg.output_dir)
print(cfg.total_steps)
print(cfg.shard_dir)
" "$@")
OUTPUT_DIR=$(echo "$CFG_INFO" | sed -n '1p')
TOTAL_STEPS=$(echo "$CFG_INFO" | sed -n '2p')
SHARD_DIR=$(echo "$CFG_INFO" | sed -n '3p')
echo "Resolved output_dir=$OUTPUT_DIR total_steps=$TOTAL_STEPS shard_dir=$SHARD_DIR"

# See LOCAL-SSD STAGING above. Falls back to the original (Lustre) SHARD_DIR
# untouched if $TMPDIR isn't set for some reason (e.g. a manual non-sbatch
# invocation while developing this script) rather than failing outright.
if [ -n "$TMPDIR" ]; then
    LOCAL_SHARD_DIR="$TMPDIR/shard_data"
    echo "Staging shard_dir to local SSD: $SHARD_DIR -> $LOCAL_SHARD_DIR"
    time cp -a "$SHARD_DIR" "$LOCAL_SHARD_DIR"
    TRAIN_SHARD_DIR="$LOCAL_SHARD_DIR"
else
    echo "\$TMPDIR not set -- skipping local-SSD staging, reading shard_dir directly from $SHARD_DIR"
    TRAIN_SHARD_DIR="$SHARD_DIR"
fi

latest_checkpoint() {
    ls "$1"/step_*.pt 2>/dev/null | sed -E 's#.*/step_([0-9]+)\.pt#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-
}

BEFORE_CKPT=$(latest_checkpoint "$OUTPUT_DIR")
if [ -n "$BEFORE_CKPT" ]; then
    BEFORE_STEP=$(basename "$BEFORE_CKPT" | sed -E 's/step_([0-9]+)\.pt/\1/')
else
    BEFORE_STEP=0
fi

# Single process for one GPU, torchrun for more than one. --shard-dir is
# appended AFTER "$@" so it wins over whatever the original args/config file
# set (see configs/README.md's own "a flag passed explicitly on the command
# line always overrides the same key in the YAML file" precedence rule) --
# every rank reads from the same staged local copy, not the original
# SHARD_DIR.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

# Shared between the normal post-`wait` path below and the SIGTERM trap
# (see --signal=B:TERM@180 above) -- identical logic either way, just
# invoked from two different places depending on WHY training stopped (a
# genuine finish/crash vs. an imminent time-limit kill). See AUTO-RESUBMIT
# above for why the trap path exists at all.
check_and_resubmit() {
    if [ -f "$OUTPUT_DIR/final.pt" ]; then
        echo "Training complete -- reached total_steps=$TOTAL_STEPS, final.pt written."
        exit 0
    fi

    echo "final.pt not found in $OUTPUT_DIR -- checking for progress to resume from."
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
    # Preserve THIS job's own --cpus-per-task too, not just --gres/--time --
    # this cluster ties memory to cpus-per-task at ~1.95GB/core (see
    # jobs/train_tokenizer/parity_bpe.sh's own comment), and a multi-GPU run needs more
    # than the script's #SBATCH --cpus-per-task=16 default (8 GPUs x
    # TrainConfig.num_workers=4 DataLoader workers = 32 processes alone).
    # Without this, a run launched with an explicit --cpus-per-task override
    # loses it on every resubmit after the first, silently reverting to 16
    # and risking an OOM on the next checkpoint load.
    CPUS_PER_TASK=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'CPUs/Task=\K\S+')
    # Same reasoning for --partition: this script's own #SBATCH pragma
    # defaults to gpu_a100_il, so a run submitted with e.g.
    # --partition=gpu_h100 would otherwise silently fall back to
    # gpu_a100_il on every resubmit after the first -- a real correctness
    # gap (H100-vs-A100 wall-clock/memory differences and gpu_a100_il's own
    # shared-pool node-type uncertainty are exactly why a partition gets
    # chosen deliberately in the first place), not just a preference lost.
    PARTITION=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Partition=\K\S+')

    echo "Progress made this run: step $BEFORE_STEP -> $AFTER_STEP (of $TOTAL_STEPS). Resubmitting from $AFTER_CKPT..."
    sbatch --partition="$PARTITION" --gres=gpu:"$NUM_GPUS" --time="$TIME_LIMIT" --cpus-per-task="$CPUS_PER_TASK" jobs/pretrain/pretraining.sh "${ORIG_ARGS[@]}" --resume-from "$AFTER_CKPT"
    SBATCH_EXIT=$?
    if [ "$SBATCH_EXIT" -ne 0 ]; then
        echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
        echo "  sbatch --partition=$PARTITION --gres=gpu:$NUM_GPUS --time=$TIME_LIMIT --cpus-per-task=$CPUS_PER_TASK jobs/pretrain/pretraining.sh ${ORIG_ARGS[*]} --resume-from $AFTER_CKPT" >&2
        exit 1
    fi
    echo "Resubmitted successfully."
    exit 0
}

# See --signal=B:TERM@180 / AUTO-RESUBMIT above: SIGTERM here means SLURM's
# hard kill is ~180s away. Forward it to the actual training process
# (SLURM's "B:" signal flag targets only this wrapper script, not children
# it spawned directly) so it stops promptly -- torchrun propagates SIGTERM
# to its own worker processes on receiving one, same as a plain single-GPU
# python process stopping directly -- then run the exact same
# check-and-resubmit this script would run on a normal exit, inside the
# grace window, before the real kill lands.
on_term() {
    echo "Received SIGTERM ($((180))s from --signal=B:TERM@180) -- stopping training and resubmitting early."
    kill -TERM "$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
    check_and_resubmit
}
trap on_term TERM

echo "Starting pretraining with $NUM_GPUS GPU(s), args: $@ --shard-dir $TRAIN_SHARD_DIR"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m systems.pretraining.cli "$@" --shard-dir "$TRAIN_SHARD_DIR" &
else
    python3 -m systems.pretraining.cli "$@" --shard-dir "$TRAIN_SHARD_DIR" &
fi
CHILD_PID=$!
wait "$CHILD_PID"
check_and_resubmit
