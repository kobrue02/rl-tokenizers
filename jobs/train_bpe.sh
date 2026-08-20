#!/bin/bash
#SBATCH --job-name=bpe_train
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Standard byte-level BPE baseline -- wraps HuggingFace's `tokenizers` directly
# (see bpe/model.py). CPU-only, Rust-backed fit (not gradient descent); 1h is a
# generous margin -- benchmarked at 0.25s for vocab_size=5000/~320KB.
# Usage: sbatch jobs/train_bpe.sh --data-source all --langs all --vocab-size 50000
# Requires HF_TOKEN (flores_plus/bouquet are gated).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

module load devel/python/3.13.3-llvm-19.1

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN -- run \`huggingface-cli login\` or export HF_TOKEN}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

# .json, not .pt -- bpe/inference.py loads via tokenizers.Tokenizer.from_file(), not torch.load.
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/bpe_${SLURM_JOB_ID}.json"
echo "Starting BPE fitting with args: $@"
python3 train.py bpe \
    --use-wandb \
    --wandb-project bpe \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/bpe_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/bpe_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Fitting complete."
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh bpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/bpe_comparison.json" --result-key bpe
else
    echo "Fitting failed." && exit 1
fi
