#!/bin/bash
#SBATCH --job-name=fanta_train
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

# FANTA: MantaModel architecture + next-byte CE plus a differentiable Gini
# penalty over per-language compression rate and a per-language rate anchor
# (the anchor exists because the Gini term alone has a degenerate "equally
# uncompressed" solution -- confirmed empirically). Batching is GROUP-based
# (fanta/train.py), unlike jobs/train_tokenizer/manta.sh's flat sampling, since both
# loss terms need several languages' rates in one forward pass.
# Usage: sbatch jobs/train_tokenizer/fanta.sh --data-source all --langs all --max-steps 20000 --vocab-size 50000
#   --lambda-fair/--lambda-rate reweight the two loss terms; --target-rate-anchor/--anchor-lang change the rate target.
# Requires HF_TOKEN (flores_plus/bouquet are gated).
#
# RESULT_KEY (env var, default "fanta"): the post-training auto-eval below
# ALWAYS runs on success and writes results/${RESULT_KEY}_comparison.json
# under --result-key "$RESULT_KEY" -- for a lambda_fair/lambda_rate
# ABLATION run (see configs/train_tokenizer/fanta_ablation_*_50k.yml), set
# this to something other than the default, or it will silently OVERWRITE
# the real fanta run's own completed results/fanta_comparison.json:
#   RESULT_KEY=fanta_ablation_anchor_only sbatch jobs/train_tokenizer/fanta.sh \
#       -c configs/train_tokenizer/fanta_ablation_anchor_only_50k.yml

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep/pretraining_data.sh

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

RESULT_KEY="${RESULT_KEY:-fanta}"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/${RESULT_KEY}_${SLURM_JOB_ID}.pt"
echo "Starting FANTA training (RESULT_KEY=${RESULT_KEY}) with args: $@"
python3 train.py fanta \
    --use-wandb \
    --wandb-project fanta \
    --run-name "${RESULT_KEY}-slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/${RESULT_KEY}_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/${RESULT_KEY}_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/eval/evaluate.sh fanta --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/${RESULT_KEY}_comparison.json" --result-key "$RESULT_KEY"
else
    echo "Training failed." && exit 1
fi
