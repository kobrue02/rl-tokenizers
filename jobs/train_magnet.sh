#!/bin/bash
#SBATCH --job-name=magnet_train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# MAGNET baseline tokenizer training -- see jobs/train.sh for the fairtok RL policy
# this is a comparison baseline for. Its training is plain differentiable backprop
# (Gumbel-sigmoid straight-through boundaries), not RL -- no reward/fairness-refresh
# machinery -- but it DOES have the same --use-wandb/--wandb-project/--run-name
# flags fairtok's job does (see magnet/train.py's MagnetConfig).
#
# Usage:
#   sbatch jobs/train_magnet.sh --data-source all --langs all --max-steps 20000 --vocab-size 50000
#   sbatch jobs/train_magnet.sh --data-source oldi_seed --max-steps 2000   # quicker, single source
#
# All train.py magnet / magnet.cli flags are forwarded directly -- see
# `python train.py magnet --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see common/oldi_data.py,
# reused here for data loading). Same HF_TOKEN handling as jobs/train.sh -- see below.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
# The cuda module above puts its OWN (older) cuDNN on LD_LIBRARY_PATH, which shadows
# the newer cuDNN PyTorch already bundles for itself (the nvidia-cudnn-cu12 wheel in
# .venv) -- if the two disagree, a cuDNN-using op fails with "PyTorch was compiled
# against X but found runtime version Y" the moment it first runs on a CUDA tensor
# (see jobs/train_manta.sh, which hits this via its block-level GRU). PyTorch
# doesn't need the module's CUDA toolkit at RUNTIME (only nvcc, which nothing here
# calls), so unset this and let PyTorch fall back to its own bundled libraries.
unset LD_LIBRARY_PATH

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN and no cached login at ~/.cache/huggingface/token -- run \`huggingface-cli login\` on this cluster (not just your laptop) or export HF_TOKEN before submitting}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

# 5. Run -- job-id-tagged output paths, same convention as jobs/train.sh
echo "Starting MAGNET training with args: $@"
python3 train.py magnet \
    --use-wandb \
    --wandb-project magnet \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$PROJECT_ROOT/checkpoints/magnet_${SLURM_JOB_ID}.pt" \
    --vocab-out "$PROJECT_ROOT/vocab_out/magnet_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/magnet_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
else
    echo "Training failed." && exit 1
fi
