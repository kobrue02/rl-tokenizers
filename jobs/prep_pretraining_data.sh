#!/bin/bash
#SBATCH --job-name=pretrain_data_prep
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Offline tokenization: stream a corpus from common/data/corpora.py's shared
# registry -- the SAME registry common/data/cli_data.py's tokenizer-training side
# uses (oldi_seed/flores_dev/smol/glot500/fineweb_edu/olmo_mix/ccmatrix/
# un_pc/europarl/tatoeba_mt/bible_nlp, no separate pretraining-only source
# list) -- tokenize with an already-trained systems/
# checkpoint, pack into token shards. See pretraining/data_prep.py's own
# module docstring for the full design and its PERFORMANCE section for a
# real, stated caveat: the five neural (span-family) tokenizer systems
# encode one document at a time, no batching, and fairtok specifically
# loops one byte at a time in Python -- expect this to be substantially
# slower than bpe/superbpe for a genuinely large prep run.
#
# No GPU requested: this is CPU-bound (network streaming + per-document
# tokenization), matching jobs/evaluate.sh's own cpu_il choice for the same
# reason. --time=12:00:00 is a starting estimate, not a benchmark at real
# scale (only tested locally against a single small Glot500 language) --
# widen it if a --langs/--dataset-config/--max-tokens combination needs longer.
#
# AUTO-RESUBMIT: mirrors jobs/train_pretraining.sh's own -- a run whose
# --max-tokens exceeds this job's own --time limit will get killed mid-run,
# well before pretraining.data_prep.prep_dataset ever gets to write
# shards_meta.json. See pretraining/data_prep.py's own RESUME docstring
# section: prep_dataset checkpoints its own progress periodically and
# resumes automatically (no extra flag) whenever it's rerun against the
# same --output-dir. This script checks after each run:
#   - shards_meta.json present in output_dir -> genuinely done, exit 0.
#   - not present, but prep_checkpoint.json's total_tokens advanced past
#     what it was before this run started -> real progress was made;
#     resubmit the exact same command via sbatch (no --resume flag needed,
#     prep_dataset detects the checkpoint on its own).
#   - not present, and no progress beyond this run's own starting point ->
#     treated as a genuine failure (e.g. a persistent crash before the
#     first checkpoint), NOT resubmitted, so a real bug can't spin into an
#     infinite resubmission loop burning allocation.
#
# Usage:
#   sbatch jobs/prep_pretraining_data.sh --dataset glot500 --langs all \
#       --system bpe --checkpoint checkpoints/bpe_12345.json \
#       --output-dir pretrain_data/glot500_bpe --max-tokens 5000000000
#
#   sbatch jobs/prep_pretraining_data.sh --dataset fineweb_edu --dataset-config sample-10BT \
#       --system superbpe --checkpoint checkpoints/superbpe_12345.pt \
#       --output-dir pretrain_data/fineweb_superbpe
#
#   sbatch jobs/prep_pretraining_data.sh --dataset glot500 --langs all \
#       --system fanta --checkpoint checkpoints/fanta_12345.pt --vocab-json vocab_out/fanta_vocab_12345.json \
#       --output-dir pretrain_data/glot500_fanta --max-tokens 500000000
#
#   sbatch jobs/prep_pretraining_data.sh --dataset oldi_seed --langs all \
#       --system fanta --checkpoint checkpoints/fanta_12345.pt --vocab-json vocab_out/fanta_vocab_12345.json \
#       --output-dir pretrain_data/oldi_fanta
#   # ^ a genuinely parallel source used for pretraining instead of the usual
#   # monolingual choices -- small relative to fineweb_edu/olmo_mix/glot500,
#   # but exactly as available now that both halves of this project share one
#   # registry: no source is off-limits to either consumer.
#
# All flags forward directly to pretraining/data_prep.py -- see
# `python3 -m pretraining.data_prep --help`.
#
# PREREQUISITE: fineweb-edu/olmo-mix/glot500 are all public but sizeable HF
# datasets -- same HF_TOKEN handling as jobs/train.sh (a token isn't
# strictly required for these three specifically, unlike flores_plus/
# bouquet, but is set anyway since a logged-in session avoids anonymous
# rate limits on a long streaming run).

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: a much larger-quota Lustre workspace (44TiB vs $HOME's 550GiB
# hard cap that caused this exact job's 2026-08 EDQUOT crash) -- allocated
# via `ws_allocate rl-tokenizers 60`. NOT permanent: it expires unless
# renewed with `ws_extend rl-tokenizers <n>`. Only regenerable/derived data
# (HF cache, prep output) lives here -- the actual repo/code stays under
# $PROJECT_ROOT on $HOME, which is small and permanent.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 1.5. Preflight: fail fast if the Lustre quota on this filesystem is
# already maxed out, instead of burning hours of tokenization before a
# partial shard write dies with a cryptic short-write OSError (real
# incident: this exact job ate ~75min before an EDQUOT-driven partial
# write killed the run -- `lfs quota` showed the user was already ~18MB
# over the 550GiB hard limit before the job even started). Skips silently
# if `lfs` isn't present (non-Lustre cluster) rather than failing the job
# over an unrelated environment difference.
if command -v lfs >/dev/null 2>&1; then
    QUOTA_MOUNT=$(df --output=target "$PROJECT_ROOT" 2>/dev/null | tail -1)
    QUOTA_LINE=$(lfs quota -u "$(whoami)" "$QUOTA_MOUNT" 2>/dev/null | awk 'NR==3')
    QUOTA_KBYTES=$(awk '{print $2}' <<< "$QUOTA_LINE" | tr -d '*')
    QUOTA_BLIMIT=$(awk '{print $4}' <<< "$QUOTA_LINE")
    if [[ "$QUOTA_KBYTES" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" -gt 0 ]]; then
        QUOTA_PCT=$((QUOTA_KBYTES * 100 / QUOTA_BLIMIT))
        echo "Preflight: Lustre quota on $QUOTA_MOUNT at ${QUOTA_PCT}% (${QUOTA_KBYTES}K / ${QUOTA_BLIMIT}K hard limit)."
        if [ "$QUOTA_PCT" -ge 95 ]; then
            echo "Quota on $QUOTA_MOUNT is at ${QUOTA_PCT}% -- too close to the hard limit to safely start this run. Free space (du -h --max-depth=1 under \$HOME) or move heavy dirs to a larger workspace before resubmitting." >&2
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
mkdir -p logs pretrain_data

# 5. Resolve this run's output_dir from the EXACT args this job received
# (config file + any prior state) -- reuses pretraining.data_prep's own
# parsing so this can never drift from what it actually uses.
OUTPUT_DIR=$(python3 -c "
import sys
from pretraining.data_prep import build_arg_parser
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

# 6. Run
echo "Starting pretraining data prep with args: $@"
python3 -m pretraining.data_prep "$@"
PREP_EXIT=$?

# 7. Done, or resubmit? See the AUTO-RESUBMIT comment at the top.
if [ -f "$OUTPUT_DIR/shards_meta.json" ]; then
    echo "Data prep complete."
    exit 0
fi

echo "shards_meta.json not found in $OUTPUT_DIR (exited with code $PREP_EXIT) -- checking for progress to resume from."
AFTER_TOKENS=$(checkpoint_tokens "$CKPT_PATH")
if [ "$AFTER_TOKENS" -le "$BEFORE_TOKENS" ]; then
    echo "No progress made this run (before=$BEFORE_TOKENS after=$AFTER_TOKENS tokens) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi

# This job's OWN actual --time limit (which may have been overridden at
# submission, e.g. `sbatch --time=72:00:00 ...` -- see configs/prep_bpe_large.yml's
# own real OOM/timeout history) -- a bare resubmit would otherwise silently
# fall back to this script's #SBATCH --time=12:00:00 default, the same
# class of bug train_pretraining.sh's own --gres preservation guards
# against.
TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

echo "Progress made this run: $BEFORE_TOKENS -> $AFTER_TOKENS tokens. Resubmitting..."
sbatch --time="$TIME_LIMIT" jobs/prep_pretraining_data.sh "$@"
SBATCH_EXIT=$?
if [ "$SBATCH_EXIT" -ne 0 ]; then
    echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
    echo "  sbatch --time=$TIME_LIMIT jobs/prep_pretraining_data.sh $@" >&2
    exit 1
fi
echo "Resubmitted successfully."
exit 0
