#!/bin/bash
#SBATCH --job-name=superbpe_train
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# SuperBPE baseline (Liu et al. 2025, COLM) -- from-scratch pure-Python
# reimplementation of the two-stage algorithm (see superbpe/model.py), not a
# port of the official Rust trainer. CPU-only, single-shot fit.
# --checkpoint-dir (FIT_CHECKPOINT_DIR below) is a FIXED path so a resubmit
# after a timeout resumes the same in-progress fit deterministically -- only
# `rm -rf` it if starting a genuinely different experiment (different
# corpus/vocab-size); reusing it across configs raises a loud error rather
# than silently corrupting the fit.
# Usage: sbatch jobs/train_superbpe.sh --data-source all --langs all --vocab-size 50000
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

CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/superbpe_${SLURM_JOB_ID}.pt"
FIT_CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/superbpe_fit_checkpoint"
echo "Starting SuperBPE fitting with args: $@"
python3 train.py superbpe \
    --use-wandb \
    --wandb-project superbpe \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --checkpoint-dir "$FIT_CHECKPOINT_DIR" \
    --vocab-out "$PROJECT_ROOT/vocab_out/superbpe_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/superbpe_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Fitting complete."
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh superbpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/superbpe_comparison.json" --result-key superbpe
else
    echo "Fitting failed." && exit 1
fi
