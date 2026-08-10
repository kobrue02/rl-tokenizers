#!/bin/bash
#SBATCH --job-name=manta_train
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

# MANTa baseline tokenizer training -- see jobs/train.sh for the fairtok RL policy
# this is a comparison baseline for. Unlike that job, MANTa has no
# --use-wandb/--wandb-* flags (its Trainer has no wandb integration -- see
# manta/train.py) and its training is plain next-byte cross-entropy backprop through
# a Gaussian-relaxed soft byte-to-block assignment -- no reward, no boundary-rate
# loss, no fairness-refresh machinery at all (see manta/model.py's module docstring
# for why MANTa genuinely has none of that, not just a simplified version of it).
#
# Usage:
#   sbatch jobs/train_manta.sh --data-source all --langs all --max-steps 20000 --vocab-size 50000
#   sbatch jobs/train_manta.sh --data-source oldi_seed --max-steps 2000   # quicker, single source
#
# All train_manta.py / manta.cli flags are forwarded directly -- see
# `python train_manta.py --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see fairtok/oldi_data.py,
# reused here for data loading). Same HF_TOKEN handling as jobs/train.sh -- see below.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/fairtok

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

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
echo "Starting MANTa training with args: $@"
python3 train_manta.py \
    --output-dir "$PROJECT_ROOT/checkpoints/manta_${SLURM_JOB_ID}.pt" \
    --vocab-out "$PROJECT_ROOT/vocab_out/manta_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/manta_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
else
    echo "Training failed." && exit 1
fi
