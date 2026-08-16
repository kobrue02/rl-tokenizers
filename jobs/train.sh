#!/bin/bash
#SBATCH --job-name=fairtok_train
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

# Fairness-aware byte-boundary policy training.
#
# Usage:
#   sbatch jobs/train.sh --data-source all --langs all --num-train-epochs 3 --lambda-target 20.0 --vocab-size 50000
#   sbatch jobs/train.sh --data-source oldi_seed --num-train-epochs 1   # quicker, single source
#
# All train.py fairtok / fairtok.cli flags are forwarded directly -- see
# `python train.py fairtok --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see common/data/oldi_data.py).
# If you've already run `huggingface-cli login` on this cluster (not just your laptop --
# the token lives in $HOME on whichever machine you logged in on), that's picked up
# automatically below. Otherwise export HF_TOKEN before submitting:
#   export HF_TOKEN=hf_...
#   sbatch jobs/train.sh ...

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: larger-quota Lustre workspace for cache/derived data -- see
# jobs/prep_pretraining_data.sh's own WORK_ROOT comment for why. Expires
# unless renewed (`ws_extend rl-tokenizers <n>`) -- only cache/derived
# data lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
# The cuda module above puts its OWN (older) cuDNN on LD_LIBRARY_PATH, which shadows
# the newer cuDNN PyTorch already bundles for itself (the nvidia-cudnn-cu12 wheel in
# .venv) -- if the two disagree, a cuDNN-using op (e.g. manta's block-level GRU)
# fails with "PyTorch was compiled against X but found runtime version Y" the moment
# it first runs on a CUDA tensor. PyTorch doesn't need the module's CUDA toolkit at
# RUNTIME (only nvcc, which nothing here calls), so unset this and let PyTorch fall
# back to its own bundled libraries instead.
unset LD_LIBRARY_PATH

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
export HF_HOME=$WORK_ROOT/.cache/huggingface
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
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/policy_${SLURM_JOB_ID}.pt"
echo "Starting training with args: $@"
python3 train.py fairtok \
    --use-wandb \
    --wandb-project fairtok \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
    # Held-out BOUQuET DEV was already scored periodically during training (see
    # fairtok/train.py's epoch-boundary eval); this final job scores the
    # genuinely held-out TEST split exactly once, using the checkpoint training
    # just wrote -- no --dependency needed since training already finished by
    # the time this line runs.
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh fairtok --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test
else
    echo "Training failed." && exit 1
fi
