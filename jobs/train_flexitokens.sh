#!/bin/bash
#SBATCH --job-name=flexitokens_train
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

# FlexiTokens baseline -- plain differentiable backprop (no RL/reward machinery),
# comparison baseline for jobs/train.sh's fairtok policy.
# Usage: sbatch jobs/train_flexitokens.sh --data-source all --langs all --max-steps 20000 --vocab-size 50000
# Requires HF_TOKEN (flores_plus/bouquet are gated).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # avoids the cuda module's older cuDNN shadowing PyTorch's bundled one (crashes the GRU otherwise)

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN -- run \`huggingface-cli login\` or export HF_TOKEN}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/flexitokens_${SLURM_JOB_ID}.pt"
echo "Starting FlexiTokens training with args: $@"
python3 train.py flexitokens \
    --use-wandb \
    --wandb-project flexitokens \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/flexitokens_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/flexitokens_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh flexitokens --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/flexitokens_comparison.json" --result-key flexitokens
else
    echo "Training failed." && exit 1
fi
