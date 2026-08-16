#!/bin/bash
#SBATCH --job-name=claude_tokenizer_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation of one or more Claude models' own token
# counts, via Anthropic's public count_tokens API -- see
# systems/tokenization/claude_tokenizer/evaluate.py and model.py's own module docstrings
# for the full design. Genuinely different from every other jobs/evaluate*.sh
# in this repo: no local tokenizer, no model weights, ONE REAL NETWORK CALL
# PER (group, language) PAIR -- there is no batch endpoint. Only
# compression/fertility/token parity are reported; Rényi efficiency and
# Gini are NOT available (the public API returns just a total count, no
# per-token identities) -- see model.py's own docstring for why that's a
# real API limitation, not a bug here.
#
# No GPU requested: this is pure network I/O (rate-limited HTTP calls to
# Anthropic's API), same cpu_il reasoning as jobs/evaluate_hf_frontier.sh.
#
# --time=08:00:00: deliberately NOT sized to cover the full run -- a real
# 429 confirmed this org's actual rate limit is 100/min, not the "Start"
# tier's published 2000/min (configs/eval_claude.yml now uses rpm=90, a
# safety margin under that real number). At rpm=90, the full ~272k-pair
# bouquet_test run takes ~50 hours, which won't fit in one allocation on
# most clusters anyway. Instead: --checkpoint-dir (set in
# configs/eval_claude.yml) makes every completed call durable as it
# happens, so the intended workflow is to just resubmit this exact job
# (`sbatch jobs/evaluate_claude.sh -c configs/eval_claude.yml`) again after
# it times out -- it picks up where it left off instead of re-querying
# (and re-paying for) everything from scratch. Repeat until it reports 0
# remaining. Narrow with --num-groups / --eval-data-source bouquet (dev,
# not test) for a much cheaper exploratory pass, or raise --rpm to match a
# real rate-limit increase if you get one. One model failing (see
# systems/tokenization/claude_tokenizer/evaluate.py's own per-model error isolation)
# doesn't abort the rest of --model's list.
#
# Usage:
#   sbatch jobs/evaluate_claude.sh -c configs/eval_claude.yml
#
#   sbatch jobs/evaluate_claude.sh \
#       --model claude-opus-5 --eval-data-source bouquet_test --rpm 2000 \
#       --output results/claude_comparison.json --use-wandb --run-name claude_v1
#
# All flags forward directly to systems/tokenization/claude_tokenizer/evaluate.py -- see
# `python3 evaluate.py claude_tokenizer --help`.
#
# PREREQUISITES:
#   - A real ANTHROPIC_API_KEY (export it, or pass --api-key) -- a live
#     account credential, DIFFERENT from and in addition to HF_TOKEN below.
#     count_tokens is free to call but still requires authentication and is
#     subject to your account's own RPM limit (match --rpm to your actual
#     tier -- see systems/tokenization/claude_tokenizer/evaluate.py's own --rpm help).
#   - BOUQuET is ALSO a gated HF dataset (same HF_TOKEN requirement as
#     jobs/evaluate.sh/jobs/evaluate_hf_frontier.sh -- run
#     `huggingface-cli login` on this cluster, or export HF_TOKEN) --
#     needed for --eval-data-source bouquet/bouquet_test specifically, a
#     completely separate credential from ANTHROPIC_API_KEY above.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: larger-quota Lustre workspace for cache/derived data -- see
# jobs/prep_pretraining_data.sh's own WORK_ROOT comment for why. Expires
# unless renewed (`ws_extend rl-tokenizers <n>`) -- only cache/derived
# data lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 1.5. Preflight: fail fast if the Lustre quota on this filesystem is
# already maxed out -- see jobs/prep_pretraining_data.sh's own preflight
# comment for the real incident this guards against. Skips silently if
# `lfs` isn't present (non-Lustre cluster).
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
: "${ANTHROPIC_API_KEY:?No ANTHROPIC_API_KEY set -- export it before submitting (see the PREREQUISITES comment above)}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

# 5. Run
echo "Starting claude_tokenizer evaluation with args: $@"
python3 evaluate.py claude_tokenizer "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
