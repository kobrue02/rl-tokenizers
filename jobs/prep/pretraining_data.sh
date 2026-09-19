#!/bin/bash
#SBATCH --job-name=pretrain_data_prep
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Offline tokenization: stream a corpus (common/data/corpora.py's shared
# registry), tokenize with an already-trained systems/ checkpoint, pack into
# token shards (systems/pretraining/data_prep.py). CPU-bound (network
# streaming + tokenization), no GPU needed. The five neural (span-family)
# systems encode one document at a time (fairtok byte-by-byte) -- expect
# these to be much slower than bpe/superbpe at real scale.
#
# AUTO-RESUBMIT: mirrors jobs/pretrain/pretraining.sh's own -- resumes
# automatically from prep_checkpoint.json (no extra flag) whenever the
# underlying python process exits early -- an OOM/crash (this script's own
# `wait` below returns normally, so the check-and-resubmit logic runs
# directly), OR this job approaching its --time limit. These need
# DIFFERENT handling: a genuine SLURM TIMEOUT kills the ENTIRE job -- this
# wrapper script included, not just its python child -- the instant the
# limit is reached, so code after a plain foreground `python3 ...` call
# would never run at all (confirmed live: a real TIMEOUT exit's own log
# showed no "checking for progress"/"Resubmitted successfully" message
# whatsoever, unlike every OOM-kill case, which printed both).
# --signal=B:TERM@120 asks SLURM to instead send SIGTERM to this SCRIPT
# 120s before the hard limit -- caught by the trap below, which runs the
# exact same check-and-resubmit logic early, inside that 120s grace
# window, before SLURM's own kill actually lands.
#
# PREREQUISITE for --dataset glot500: reads from a one-time local disk cache
# (common/data/prepare_glot500.py -- run jobs/prep/glot500.sh first), not
# live HF streaming -- re-streaming its ~308GB/~411-config corpus on every
# run (and every resume) was the actual bottleneck of a real prep. Pass
# --dataset-config pointing at the cache if it's not at the default
# GLOT500_LOCAL_DIR ("data/glot500").
#
# Usage:
#   sbatch jobs/prep/pretraining_data.sh --dataset glot500 --langs all \
#       --dataset-config "$WORK_ROOT/data/glot500" \
#       --system bpe --checkpoint checkpoints/bpe_12345.json \
#       --output-dir pretrain_data/glot500_bpe --max-tokens 5000000000
#   sbatch jobs/prep/pretraining_data.sh --dataset fineweb_edu --dataset-config sample-10BT \
#       --system superbpe --checkpoint checkpoints/superbpe_12345.pt \
#       --output-dir pretrain_data/fineweb_superbpe
#
# All flags forward directly to systems/pretraining/data_prep.py -- see
# `python3 -m systems.pretraining.data_prep --help`.
# Requires HF_TOKEN (avoids anonymous rate limits on a long streaming run).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# Captured HERE, before anything else can consume/shift "$@" -- BASH FUNCTIONS
# have their OWN positional-parameter scope, separate from the top-level
# script's, unless explicitly forwarded (e.g. `foo "$@"`). check_and_resubmit()
# below is always called BARE (no args) from both the normal post-`wait` path
# and the on_term SIGTERM trap, so a literal "$@" inside it is ALWAYS empty --
# confirmed live: every auto-resubmit was silently dropping -c config.yml and
# resubmitting with bare argparse defaults (missing required value(s) for
# ['dataset', 'system', 'checkpoint', 'output_dir']). ORIG_ARGS is a real
# array, not a string, so later use as "${ORIG_ARGS[@]}" preserves exact
# argument boundaries the same way "$@" would.
ORIG_ARGS=("$@")

# 44TiB Lustre scratch (`ws_allocate rl-tokenizers 60`, expires unless
# `ws_extend`'d) vs $HOME's 550GiB hard cap -- only regenerable/derived data
# lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# Preflight: fail fast if $PROJECT_ROOT's Lustre quota is already maxed out
# instead of burning hours before a partial write dies with a cryptic
# short-write OSError (real incident: a run ate ~75min before an
# EDQUOT-driven partial write killed it, ~18MB over the 550GiB limit before
# the job even started). Skips silently if `lfs` isn't present.
if command -v lfs >/dev/null 2>&1; then
    QUOTA_MOUNT=$(df --output=target "$PROJECT_ROOT" 2>/dev/null | tail -1)
    QUOTA_LINE=$(lfs quota -u "$(whoami)" "$QUOTA_MOUNT" 2>/dev/null | awk 'NR==3')
    QUOTA_KBYTES=$(awk '{print $2}' <<< "$QUOTA_LINE" | tr -d '*')
    QUOTA_BLIMIT=$(awk '{print $4}' <<< "$QUOTA_LINE")
    if [[ "$QUOTA_KBYTES" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" -gt 0 ]]; then
        QUOTA_PCT=$((QUOTA_KBYTES * 100 / QUOTA_BLIMIT))
        echo "Preflight: Lustre quota on $QUOTA_MOUNT at ${QUOTA_PCT}% (${QUOTA_KBYTES}K / ${QUOTA_BLIMIT}K hard limit)."
        if [ "$QUOTA_PCT" -ge 95 ]; then
            echo "Quota on $QUOTA_MOUNT is at ${QUOTA_PCT}% -- too close to the hard limit to safely start this run. Free space or move heavy dirs to a larger workspace before resubmitting." >&2
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
mkdir -p logs pretrain_data

# Resolve output_dir from the exact args this job received -- reuses
# systems.pretraining.data_prep's own parsing so this can't drift from it.
OUTPUT_DIR=$(python3 -c "
import sys
from systems.pretraining.data_prep import build_arg_parser
from common.config_file import parse_args_with_config
args = parse_args_with_config(build_arg_parser(), sys.argv[1:])
print(args.output_dir)
" "$@")
CKPT_PATH="$OUTPUT_DIR/prep_checkpoint.json"

checkpoint_tokens() {
    python3 -c "
import json
try:
    with open('$1') as f:
        print(json.load(f).get('total_tokens', 0))
except FileNotFoundError:
    print(0)
"
}
BEFORE_TOKENS=$(checkpoint_tokens "$CKPT_PATH")

# Shared between the normal post-`wait` path below AND the SIGTERM trap
# (see --signal=B:TERM@120 above) -- identical logic either way, just
# invoked from two different places depending on WHY the python process
# stopped (a genuine finish/crash vs. an imminent time-limit kill).
check_and_resubmit() {
    if [ -f "$OUTPUT_DIR/shards_meta.json" ]; then
        echo "Data prep complete."
        exit 0
    fi

    echo "shards_meta.json not found in $OUTPUT_DIR -- checking for progress to resume from."
    AFTER_TOKENS=$(checkpoint_tokens "$CKPT_PATH")
    if [ "$AFTER_TOKENS" -le "$BEFORE_TOKENS" ]; then
        echo "No progress made this run (before=$BEFORE_TOKENS after=$AFTER_TOKENS tokens) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
        exit 1
    fi

    TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

    echo "Progress made this run: $BEFORE_TOKENS -> $AFTER_TOKENS tokens. Resubmitting..."
    sbatch --time="$TIME_LIMIT" jobs/prep/pretraining_data.sh "${ORIG_ARGS[@]}"
    SBATCH_EXIT=$?
    if [ "$SBATCH_EXIT" -ne 0 ]; then
        echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
        echo "  sbatch --time=$TIME_LIMIT jobs/prep/pretraining_data.sh ${ORIG_ARGS[*]}" >&2
        exit 1
    fi
    echo "Resubmitted successfully."
    exit 0
}

# See --signal=B:TERM@120 / AUTO-RESUBMIT above: SIGTERM here means SLURM's
# hard kill is ~120s away. Forward it to the actual python process (SLURM's
# "B:" signal flag targets only this wrapper script, not children it
# spawned directly) so it stops promptly, then run the exact same
# check-and-resubmit this script would run on a normal exit -- inside the
# grace window, before the real kill lands.
on_term() {
    echo "Received SIGTERM ($((120))s from --signal=B:TERM@120) -- stopping the tokenizer and resubmitting early."
    kill -TERM "$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
    check_and_resubmit
}
trap on_term TERM

echo "Starting pretraining data prep with args: $@"
python3 -m systems.pretraining.data_prep "$@" &
CHILD_PID=$!
wait "$CHILD_PID"
check_and_resubmit
