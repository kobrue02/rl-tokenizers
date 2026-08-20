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

# One-time local prep for cis-lmu/Glot500 (common/data/prepare_glot500.py) --
# run ONCE before --dataset glot500 is usable (no live fallback). Unlike
# prepare_bible_nlp.sh: resumable per-language and auto-resubmitting (~308GB
# across ~411 languages won't finish in one SLURM window).
# AUTO-RESUBMIT measured in "how many languages' .jsonl files exist": done
# when languages-done >= requested; resubmits if more finished than at this
# run's start; treated as a real failure (not resubmitted) otherwise.
#
# Usage:
#   sbatch jobs/prepare_glot500.sh --output-dir "$WORK_ROOT/data/glot500" --limit 5 --max-workers 4  # smoke test first
#   sbatch jobs/prepare_glot500.sh --output-dir "$WORK_ROOT/data/glot500" --max-workers 8             # full run
# NOTE: $WORK_ROOT isn't expanded until this script runs -- pass --output-dir
# literally at submission time (SLURM won't expand your shell's own vars).
# All flags forward directly -- see `python3 -m common.data.prepare_glot500 --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # same 44TiB scratch every jobs/*.sh uses -- see jobs/prep_pretraining_data.sh

# Preflight: fail fast if WORK_ROOT's own quota is maxed out (this ~308GB
# cache lands there, unlike jobs/prep_pretraining_data.sh's PROJECT_ROOT check).
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

module load devel/python/3.13.3-llvm-19.1

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

# Resolve output_dir and total requested language count from the exact args
# this job received -- reuses prepare_glot500's own parsing so this can't drift.
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

echo "Starting glot500 prep with args: $@ (already done: $BEFORE_COUNT/$TOTAL_REQUESTED languages)"
python3 -m common.data.prepare_glot500 "$@"
PREP_EXIT=$?
AFTER_COUNT=$(count_done "$OUTPUT_DIR")

if [ "$AFTER_COUNT" -ge "$TOTAL_REQUESTED" ]; then
    echo "glot500 prep complete: $AFTER_COUNT/$TOTAL_REQUESTED languages done."
    exit 0
fi

echo "$AFTER_COUNT/$TOTAL_REQUESTED languages done (exited with code $PREP_EXIT) -- checking for progress to resume from."
if [ "$AFTER_COUNT" -le "$BEFORE_COUNT" ]; then
    echo "No progress made this run (before=$BEFORE_COUNT after=$AFTER_COUNT languages) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi

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
