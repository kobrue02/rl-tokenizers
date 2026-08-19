#!/bin/bash
#SBATCH --job-name=parity_bpe_train
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Parity-aware BPE baseline tokenizer fitting -- see
# systems/tokenization/parity_bpe/model.py's module docstring: this wraps
# the OFFICIAL implementation directly (Foroutan, Meister, Paul, Niklaus,
# Ahmadi, Bosselut & Sennrich, ACL 2026, github.com/swiss-ai/parity-aware-bpe,
# vendored in systems/tokenization/parity_bpe/vendor/), per explicit user
# instruction to reuse the official code wherever possible rather than
# reimplementing the algorithm. No GPU is requested: same reasoning as
# jobs/train_superbpe.sh (single-shot, pure-Python corpus-statistics fit,
# no gradient descent).
#
# --checkpoint-dir IS resumable, unlike a naive reuse of the official code
# would be: the vendored learn_bpe/learn_bpe_moving_window are themselves
# single monolithic calls with no pause/resume hook, but
# systems/tokenization/parity_bpe/checkpointed_fit.py reimplements just
# their OUTER LOOP (reusing every one of their own sub-functions unmodified)
# with periodic checkpointing added -- see that module's own docstring for
# why a resume from it is provably BYTE-IDENTICAL to letting the fit run
# uninterrupted, not merely "comparably fair". FIXED path (not tagged by
# SLURM_JOB_ID) so resubmitting this exact job after a timeout finds the
# same in-progress fit and continues it -- same convention as
# jobs/train_superbpe.sh's own --checkpoint-dir. Only clear this directory
# if you're intentionally starting a genuinely different experiment
# (different corpus/--vocab-size/--num-global-merges/--use-moving-window) --
# reusing it across two different configs raises a loud ValueError rather
# than silently corrupting the fit, but a stale directory from an abandoned
# run still needs a manual `rm -rf` first.
#
# --time=08:00:00 is a first estimate, NOT yet benchmarked at this
# project's real --vocab-size scale -- widen it if a real run needs
# multiple resubmissions to converge; profile checkpointed_fit.fit_checkpointed
# if even repeated resumes don't.
#
# Usage:
#   sbatch jobs/train_parity_bpe.sh --data-source all --langs all --vocab-size 50000
#   sbatch jobs/train_parity_bpe.sh --data-source all --vocab-size 50000 --num-global-merges 24872  # hybrid variant
#   sbatch jobs/train_parity_bpe.sh --data-source all --vocab-size 50000 --use-moving-window        # window variant
#   sbatch jobs/train_parity_bpe.sh --data-source oldi_seed --vocab-size 2000  # quicker, single source
#
# All train.py parity_bpe / parity_bpe.cli flags are forwarded directly --
# see `python train.py parity_bpe --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see
# common/data/oldi_data.py, reused here for data loading -- BOUQuET dev
# specifically doubles as parity_bpe's own REQUIRED fairness dev-set, not
# just periodic reporting like every other tokenizer here, see
# systems/tokenization/parity_bpe/train.py's own module docstring). Same
# HF_TOKEN handling as jobs/train.sh -- see below.

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
mkdir -p logs checkpoints vocab_out results

# 5. Run -- job-id-tagged output paths, same convention as jobs/train_superbpe.sh.
# CHECKPOINT_PATH (the FINAL saved model) is a .json file (tokenizers.Tokenizer's
# own native serialization -- see model.py's own docstring for why this is
# `tokenizers`-backed like bpe/, not torch.save-backed like superbpe/), not
# a .pt file. FIT_CHECKPOINT_DIR (the IN-PROGRESS fit's own resumable state,
# see checkpointed_fit.py) is deliberately NOT job-id-tagged -- see the
# --checkpoint-dir comment above for why it needs to stay fixed across
# resubmissions.
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
    # Held-out BOUQuET DEV was already scored once, right after fitting (see
    # parity_bpe/train.py's module docstring -- BOUQuET dev drove the fit
    # ITSELF here, unlike superbpe, so this is a report on data the fit
    # already saw, not held-out in the usual sense); this final job scores
    # the genuinely held-out TEST split exactly once, using the checkpoint
    # fitting just wrote. --output/--result-key included explicitly --
    # confirmed live (see jobs/train_superbpe.sh's own history) that
    # omitting them means the eval job only prints a report and writes NO
    # mergeable JSON, silently losing the ability to fold this into
    # results/all_tokenizers_comparison.json without a manual re-run.
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh parity_bpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/parity_bpe_comparison.json" --result-key parity_bpe
else
    echo "Fitting failed." && exit 1
fi
