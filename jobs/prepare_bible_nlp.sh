#!/bin/bash
#SBATCH --job-name=prepare_bible_nlp
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# One-time local prep for bible-nlp/biblenlp-corpus -- see
# common/data/prepare_bible_nlp.py's own module docstring for the full
# design (streams the ~5.2GB corpus.json ONCE, picks one canonical
# translation per language, writes local JSONL + metadata.json). Run this
# ONE TIME before --data-source bible_nlp is usable at all -- it has no
# live fallback any more (see common/data/corpora.py's own bible_nlp
# loader, which now reads this script's own output directory).
#
# --time=24:00:00 is a starting estimate, not a benchmark at real scale
# (only verified locally against a --limit 5 subset) -- this downloads the
# entire 5.2GB file in one sequential pass and parses/writes every one of
# its ~1000+ languages, so widen this if it times out partway; the script
# itself is NOT resumable (a partial run's already-written {lang}.jsonl
# files are still individually valid, but metadata.json is only written
# at the very end, so a killed run has no record of what it actually
# finished -- re-running from scratch is the safe recovery, not resuming).
#
# CPU-only, no GPU needed (network streaming + JSON parsing), matching
# jobs/prep_pretraining_data.sh's own cpu_il choice for the same reason.
#
# Usage:
#   sbatch jobs/prepare_bible_nlp.sh --output-dir data/bible_nlp
#
#   sbatch jobs/prepare_bible_nlp.sh --output-dir data/bible_nlp_test --limit 20
#   # ^ a quick subset run to sanity-check the pipeline before committing to
#   # the full multi-hour job above
#
# All flags forward directly to common/data/prepare_bible_nlp.py -- see
# `python3 -m common.data.prepare_bible_nlp --help`.
#
# PREREQUISITE: bible-nlp/biblenlp-corpus is a public (non-gated) HF
# dataset -- HF_TOKEN isn't strictly required, but is set anyway (same as
# jobs/prep_pretraining_data.sh) since a logged-in session avoids anonymous
# rate limits on a long streaming download.

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
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs data

# 5. Run
echo "Starting bible_nlp prep with args: $@"
python3 -m common.data.prepare_bible_nlp "$@"

if [ $? -eq 0 ]; then
    echo "bible_nlp prep complete."
else
    echo "bible_nlp prep failed." && exit 1
fi
