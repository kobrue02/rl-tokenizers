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

# SuperBPE baseline tokenizer fitting -- see superbpe/model.py's module docstring
# for the two-stage byte-level BPE algorithm (Liu et al. 2025, COLM) and why this
# is a from-scratch pure-Python reimplementation of the ALGORITHM, not a port of
# the official release's Rust-backed trainer. No GPU is requested: this is a
# single-shot, pure-Python corpus-statistics fit (no gradient descent, no forward
# pass), so cpu_il (this cluster's CPU-only counterpart to jobs/train*.sh's
# gpu_a100_il partition) is the correct choice -- a GPU would sit completely idle.
#
# --time=08:00:00: fitting is O(merges * corpus size) in the worst case (see
# superbpe/model.py's SCALE-DOWN NOTICE) -- a smoke-test-scale --vocab-size (a
# few hundred) finishes in seconds, but a real --vocab-size 50000 run over the
# full pooled corpus needs tens of thousands of merges and will take
# substantially longer wall-clock than the neural baselines' own GPU training.
# Confirmed live: an 8h run got through all of stage1 but died partway into
# stage2 (whole-sentence sequences cost more per merge than stage1's
# pretoken-level ones) -- --checkpoint-dir below now makes that resumable
# instead of a total loss, but --time may still need widening for a full
# --vocab-size 50000 run; profile _fit_merges if even repeated resumes don't
# converge.
#
# --checkpoint-dir is a FIXED path (not tagged by SLURM_JOB_ID, unlike
# CHECKPOINT_PATH below) so resubmitting this exact job after a timeout finds
# the same in-progress fit and continues it -- see fit_superbpe/_fit_merges's
# own docstrings. Resuming reproduces the EXACT same result as an uninterrupted
# run (fully deterministic fit, no seed). Only clear this directory if you're
# intentionally starting a genuinely different experiment (different corpus/
# --vocab-size) -- reusing it across two different configs raises a loud
# ValueError rather than silently corrupting the fit, but a stale directory
# from an abandoned run still needs a manual `rm -rf` first.
#
# Usage:
#   sbatch jobs/train_superbpe.sh --data-source all --langs all --vocab-size 50000
#   sbatch jobs/train_superbpe.sh --data-source oldi_seed --vocab-size 2000   # quicker, single source
#
# All train.py superbpe / superbpe.cli flags are forwarded directly -- see
# `python train.py superbpe --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see common/data/oldi_data.py,
# reused here for data loading). Same HF_TOKEN handling as jobs/train.sh -- see below.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: larger-quota Lustre workspace for cache/derived data -- see
# jobs/prep_pretraining_data.sh's own WORK_ROOT comment for why. Expires
# unless renewed (`ws_extend rl-tokenizers <n>`) -- only cache/derived
# data lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 2. Modules
module load devel/python/3.13.3-llvm-19.1

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN and no cached login at ~/.cache/huggingface/token -- run \`huggingface-cli login\` on this cluster (not just your laptop) or export HF_TOKEN before submitting}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

# 5. Run -- job-id-tagged output paths, same convention as jobs/train.sh.
# FIT_CHECKPOINT_DIR is deliberately NOT job-id-tagged -- see the --checkpoint-dir
# comment above for why it needs to stay fixed across resubmissions.
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
    # Held-out BOUQuET DEV was already scored once, right after fitting (see
    # superbpe/train.py's module docstring -- there is no per-epoch loop to
    # score at, unlike the neural baselines, so this is SuperBPE's only
    # in-training-adjacent dev check); this final job scores the genuinely
    # held-out TEST split exactly once, using the checkpoint fitting just wrote.
    # --output/--result-key included explicitly -- CONFIRMED LIVE this was
    # previously missing here, so the auto-submitted job only printed a
    # report and wrote NO mergeable JSON, silently losing the ability to
    # fold a real completed run into results/all_tokenizers_comparison.json
    # without a manual re-run (had to be redone by hand once already).
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh superbpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/superbpe_comparison.json" --result-key superbpe
else
    echo "Fitting failed." && exit 1
fi
