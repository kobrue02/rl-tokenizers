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

# Pretraining -- see pretraining/train.py's own module docstring for what's
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
# This job reuses train_manta.sh's own cuDNN workaround (see below) since
# pretraining/model.py also runs a CUDA-backed torch model on this cluster --
# same root cause (the CUDA module's own older cuDNN shadowing PyTorch's
# bundled one), already diagnosed once for MANTa's block-level GRU.
#
# Usage:
#   sbatch jobs/train_pretraining.sh --shard-dir pretrain_data/glot500_bpe \
#       --model-size small --total-steps 50000 --seq-len 1024 --per-device-batch-size 16
#
# All flags forward directly to pretraining/cli.py -- see
# `python3 -m pretraining.cli --help`.

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

# 5. Run -- single process for one GPU, torchrun for more than one.
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting pretraining with $NUM_GPUS GPU(s), args: $@"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m pretraining.cli "$@"
else
    python3 -m pretraining.cli "$@"
fi

if [ $? -eq 0 ]; then
    echo "Training complete."
else
    echo "Training failed." && exit 1
fi
