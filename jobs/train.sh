#!/bin/bash
#SBATCH --job-name=fairtok_train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Fairness-aware byte-boundary policy training.
#
# Usage:
#   sbatch jobs/train.sh --data-source all --langs all --num-epochs 3 --lambda-target 20.0 --vocab-budget 50000
#   sbatch jobs/train.sh --data-source oldi_seed --num-epochs 1   # quicker, single source
#
# All main.py / fairtok.cli flags are forwarded directly -- see `python main.py --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see fairtok/oldi_data.py).
# If you've already run `huggingface-cli login` on this cluster (not just your laptop --
# the token lives in $HOME on whichever machine you logged in on), that's picked up
# automatically below. Otherwise export HF_TOKEN before submitting:
#   export HF_TOKEN=hf_...
#   sbatch jobs/train.sh ...

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/fairtok

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

# 3. Environment
# huggingface_hub looks for the login token at $HF_HOME/token by default. We redirect
# HF_HOME below to keep this project's dataset cache out of your global ~/.cache, which
# would ALSO hide a `huggingface-cli login` token cached at the default location -- so
# pull it forward into HF_TOKEN first, before HF_HOME is overridden, if it's not already
# set explicitly.
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

# 5. Run -- job-id-tagged output paths so concurrent/sequential runs never clobber
# each other (mirrors the %x_%j convention already used for the log files above).
# Flags after these defaults come from sbatch's "$@" and can override any of them
# (argparse takes the last occurrence of a repeated flag).
echo "Starting training with args: $@"
python3 main.py \
    --use-wandb \
    --wandb-project fairtok \
    --wandb-run-name "slurm-${SLURM_JOB_ID}" \
    --checkpoint-out "$PROJECT_ROOT/checkpoints/policy_${SLURM_JOB_ID}.pt" \
    --vocab-out "$PROJECT_ROOT/vocab_out/vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
else
    echo "Training failed." && exit 1
fi
