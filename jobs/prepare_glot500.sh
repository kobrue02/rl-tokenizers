#!/bin/bash
#SBATCH --job-name=prepare_glot500
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# One-time local prep for cis-lmu/Glot500 -- see
# common/data/prepare_glot500.py's own module docstring for the full design
# (per-language streamed download, atomic temp+rename per language,
# ThreadPoolExecutor-parallelized, resumable). Run this ONE TIME before
# --dataset glot500 is usable in systems/pretraining/data_prep.py at all --
# it has no live fallback any more (see common/data/corpora.py's own
# glot500 loader, which now reads this script's own output directory).
#
# UNLIKE jobs/prepare_bible_nlp.sh: this IS resumable (rerunning the exact
# same command skips every language already fully downloaded) and IS
# auto-resubmitting (see AUTO-RESUBMIT below) -- necessary because the full
# corpus is ~308GB across ~411 languages and will not finish inside one
# SLURM time window, unlike bible_nlp's single ~5.2GB file.
#
# AUTO-RESUBMIT: mirrors jobs/train_pretraining.sh's/jobs/prep_pretraining_data.sh's
# own pattern, measured in "how many languages' .jsonl files exist" instead
# of a token/step count:
#   - languages done >= languages requested -> genuinely done, exit 0.
#   - not done, but more languages finished than were done when this run
#     started -> real progress was made; resubmit the exact same command
#     (no extra flag needed -- prepare_glot500 skips already-done languages
#     on its own).
#   - not done, and no more languages finished than this run's own starting
#     point -> treated as a genuine failure (e.g. a persistent crash before
#     even one language completed), NOT resubmitted.
#
# Usage:
#   sbatch jobs/prepare_glot500.sh --output-dir "$WORK_ROOT/data/glot500" --limit 5 --max-workers 4
#   # ^ RECOMMENDED FIRST: a quick smoke test -- concurrent streaming loads of
#   # different configs of the same HF repo haven't been exercised in this
#   # codebase before; confirm this actually works before committing to the
#   # full run below.
#
#   sbatch jobs/prepare_glot500.sh --output-dir "$WORK_ROOT/data/glot500" --max-workers 8
#   # ^ the real, full ~411-language run -- expect several resubmits.
#
# All flags forward directly to common/data/prepare_glot500.py -- see
# `python3 -m common.data.prepare_glot500 --help`. NOTE: $WORK_ROOT isn't
# expanded until step 1 below runs, so if invoking with an explicit
# --output-dir at submission time, pass it literally (SLURM does not
# expand this script's own shell variables in your `sbatch` command line).
#
# PREREQUISITE: cis-lmu/Glot500 is a public (non-gated) HF dataset --
# HF_TOKEN isn't strictly required, but is set anyway (same as
# jobs/prep_pretraining_data.sh) since a logged-in session avoids anonymous
# rate limits on a long, many-config streaming download.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: the same 44TiB ws_allocate'd Lustre workspace every other
# jobs/*.sh already uses for cache/derived data (see jobs/prep_pretraining_data.sh's
# own WORK_ROOT comment for the real EDQUOT crash history behind why this
# exists) -- confirmed as the intended destination for this ~308GB cache
# too, not a new/separate workspace.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 1.5. Preflight: fail fast if WORK_ROOT's own Lustre quota is already
# maxed out, instead of burning hours of downloading before a partial
# write dies with a cryptic short-write OSError. UNLIKE jobs/prep_pretraining_data.sh's
# own preflight (which checks PROJECT_ROOT's mount, appropriate for THAT
# script's much smaller output), this checks WORK_ROOT's mount -- that's
# where this ~308GB cache actually lands. Skips silently if `lfs` isn't
# present (non-Lustre cluster).
if command -v lfs >/dev/null 2>&1; then
    QUOTA_MOUNT=$(df --output=target "$WORK_ROOT" 2>/dev/null | tail -1)
    QUOTA_LINE=$(lfs quota -u "$(whoami)" "$QUOTA_MOUNT" 2>/dev/null | awk 'NR==3')
    QUOTA_KBYTES=$(awk '{print $2}' <<< "$QUOTA_LINE" | tr -d '*')
    QUOTA_BLIMIT=$(awk '{print $4}' <<< "$QUOTA_LINE")
    if [[ "$QUOTA_KBYTES" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" -gt 0 ]]; then
        QUOTA_PCT=$((QUOTA_KBYTES * 100 / QUOTA_BLIMIT))
        echo "Preflight: Lustre quota on $QUOTA_MOUNT (WORK_ROOT) at ${QUOTA_PCT}% (${QUOTA_KBYTES}K / ${QUOTA_BLIMIT}K hard limit)."
        if [ "$QUOTA_PCT" -ge 95 ]; then
            echo "Quota on $QUOTA_MOUNT is at ${QUOTA_PCT}% -- too close to the hard limit to safely start a ~308GB glot500 download. Check \`ws_list\` / free space before resubmitting." >&2
            exit 1
        fi
    fi
fi

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
mkdir -p logs

# 5. Resolve this run's output_dir and total requested language count from
# the EXACT args this job received -- reuses prepare_glot500's own arg
# parsing and lang_list derivation so this can never drift from what it
# actually does (see AUTO-RESUBMIT above).
read -r OUTPUT_DIR TOTAL_REQUESTED <<< "$(python3 -c "
import sys
from common.data.prepare_glot500 import build_arg_parser
from common.data.corpora import list_glot500_configs
args = build_arg_parser().parse_args(sys.argv[1:])
langs = None if args.langs == 'all' else args.langs.split(',')
lang_list = list_glot500_configs() if langs in (None, 'all') else list(langs)
if args.limit:
    lang_list = lang_list[:args.limit]
print(args.output_dir, len(lang_list))
" "$@")"

count_done() {
    ls "$1"/*.jsonl 2>/dev/null | wc -l | tr -d ' '
}
BEFORE_COUNT=$(count_done "$OUTPUT_DIR")

# 6. Run
echo "Starting glot500 prep with args: $@ (already done: $BEFORE_COUNT/$TOTAL_REQUESTED languages)"
python3 -m common.data.prepare_glot500 "$@"
PREP_EXIT=$?
AFTER_COUNT=$(count_done "$OUTPUT_DIR")

# 7. Done, or resubmit? See the AUTO-RESUBMIT comment at the top.
if [ "$AFTER_COUNT" -ge "$TOTAL_REQUESTED" ]; then
    echo "glot500 prep complete: $AFTER_COUNT/$TOTAL_REQUESTED languages done."
    exit 0
fi

echo "$AFTER_COUNT/$TOTAL_REQUESTED languages done (exited with code $PREP_EXIT) -- checking for progress to resume from."
if [ "$AFTER_COUNT" -le "$BEFORE_COUNT" ]; then
    echo "No progress made this run (before=$BEFORE_COUNT after=$AFTER_COUNT languages) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi

# This job's OWN actual --time limit (may have been overridden at
# submission) -- a bare resubmit would otherwise silently fall back to
# this script's #SBATCH --time=24:00:00 default, the same class of bug
# jobs/train_pretraining.sh's own --gres preservation guards against.
TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

echo "Progress made this run: $BEFORE_COUNT -> $AFTER_COUNT/$TOTAL_REQUESTED languages. Resubmitting..."
sbatch --time="$TIME_LIMIT" jobs/prepare_glot500.sh "$@"
SBATCH_EXIT=$?
if [ "$SBATCH_EXIT" -ne 0 ]; then
    echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
    echo "  sbatch --time=$TIME_LIMIT jobs/prepare_glot500.sh $@" >&2
    exit 1
fi
echo "Resubmitted successfully."
exit 0
