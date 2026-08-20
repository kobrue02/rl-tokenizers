#!/bin/bash
#SBATCH --job-name=parity_bpe_train
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Parity-aware BPE (Foroutan et al., ACL 2026) -- wraps the OFFICIAL
# implementation directly (vendored in systems/tokenization/parity_bpe/vendor/).
# CPU-only fit. --checkpoint-dir (FIT_CHECKPOINT_DIR) is resumable and
# byte-identical to an uninterrupted run (see checkpointed_fit.py); it's a
# FIXED path so a resubmit after timeout continues the same fit -- only
# `rm -rf` it for a genuinely different experiment (different corpus/
# vocab-size/merge settings).
# --cpus-per-task=16: CONFIRMED LIVE a real --data-source all run
# (515k sentences, 345 languages, vocab_size=50000) OOM-killed at 4 cores
# (this cluster ties memory to cpus-per-task at ~1.95GB/core) -- 16 cores is
# a generous widening, not benchmarked; profile further if it OOMs again.
# BOUQuET dev is parity_bpe's own REQUIRED fairness dev-set (drives the fit
# itself), not just periodic reporting like the other baselines.
# Usage: sbatch jobs/train_parity_bpe.sh --data-source all --langs all --vocab-size 50000
#   --num-global-merges N for the hybrid variant, --use-moving-window for the window variant.
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
mkdir -p logs checkpoints vocab_out results

CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/parity_bpe_${SLURM_JOB_ID}.json"
FIT_CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/parity_bpe_fit_checkpoint"
echo "Starting Parity-aware BPE fitting with args: $@"
python3 train.py parity_bpe \
    --use-wandb \
    --wandb-project parity_bpe \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --checkpoint-dir "$FIT_CHECKPOINT_DIR" \
    --vocab-out "$PROJECT_ROOT/vocab_out/parity_bpe_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/parity_bpe_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Fitting complete."
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh parity_bpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/parity_bpe_comparison.json" --result-key parity_bpe
else
    echo "Fitting failed." && exit 1
fi
