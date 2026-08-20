#!/bin/bash
#SBATCH --job-name=claude_tokenizer_eval
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET eval of Claude models via Anthropic's public count_tokens
# API -- one real network call per (group, language) pair, no batch endpoint.
# Only compression/fertility/token parity are reported (Rényi/Gini need
# per-token identities the public API doesn't expose). CPU-only (pure network I/O).
#
# --time=08:00:00 deliberately doesn't cover a full run -- a real 429
# confirmed this org's actual rate limit is 100/min (configs/eval_claude.yml
# uses rpm=90 as a safety margin); the full bouquet_test run takes ~50h at
# that rate. --checkpoint-dir makes every completed call durable, so just
# resubmit the same command after each timeout until it reports 0 remaining.
#
# Usage:
#   sbatch jobs/evaluate_claude.sh -c configs/eval_claude.yml
#   sbatch jobs/evaluate_claude.sh --model claude-opus-5 --eval-data-source bouquet_test --rpm 2000 \
#       --output results/claude_comparison.json --use-wandb --run-name claude_v1
# All flags forward directly -- see `python3 evaluate.py claude_tokenizer --help`.
#
# PREREQUISITES: a real ANTHROPIC_API_KEY (export it, or --api-key) --
# separate credential from HF_TOKEN below, which is also needed (BOUQuET is gated).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

# Preflight: fail fast if quota's already maxed out -- see jobs/prep_pretraining_data.sh.
if command -v lfs >/dev/null 2>&1; then
    QUOTA_MOUNT=$(df --output=target "$PROJECT_ROOT" 2>/dev/null | tail -1)
    QUOTA_LINE=$(lfs quota -u "$(whoami)" "$QUOTA_MOUNT" 2>/dev/null | awk 'NR==3')
    QUOTA_KBYTES=$(awk '{print $2}' <<< "$QUOTA_LINE" | tr -d '*')
    QUOTA_BLIMIT=$(awk '{print $4}' <<< "$QUOTA_LINE")
    if [[ "$QUOTA_KBYTES" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" =~ ^[0-9]+$ && "$QUOTA_BLIMIT" -gt 0 ]]; then
        QUOTA_PCT=$((QUOTA_KBYTES * 100 / QUOTA_BLIMIT))
        echo "Preflight: Lustre quota on $QUOTA_MOUNT at ${QUOTA_PCT}% (${QUOTA_KBYTES}K / ${QUOTA_BLIMIT}K hard limit)."
        if [ "$QUOTA_PCT" -ge 95 ]; then
            echo "Quota on $QUOTA_MOUNT is at ${QUOTA_PCT}% -- too close to the hard limit to safely start this run." >&2
            exit 1
        fi
    fi
fi

module load devel/python/3.13.3-llvm-19.1

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${ANTHROPIC_API_KEY:?No ANTHROPIC_API_KEY set -- export it before submitting}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

echo "Starting claude_tokenizer evaluation with args: $@"
python3 evaluate.py claude_tokenizer "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
