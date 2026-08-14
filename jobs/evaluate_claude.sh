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
# systems/claude_tokenizer/evaluate.py and model.py's own module docstrings
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
# --time=08:00:00: a ROUGH, UNBENCHMARKED estimate, not a measured number
# (this repo has no ANTHROPIC_API_KEY available in its own dev/test
# environment to time a real run against) -- derived from the theoretical
# floor: scoring BOUQuET test's full ~272k (group, language) pairs at
# --rpm 2000 (the "Start" tier default) takes at least 272000/2000 ≈ 136
# minutes if the rate limit is perfectly saturated the whole time; real
# runs will be slower (thread/connection overhead, occasional retries,
# network jitter), hence the wide margin here. Widen further if you use a
# lower --rpm, narrow with --num-groups / --eval-data-source bouquet (dev,
# not test) for a much cheaper exploratory pass, or raise --rpm to match a
# higher usage tier (Build=4000, Scale=8000) if you have one. One model
# failing (see systems/claude_tokenizer/evaluate.py's own per-model error
# isolation) doesn't abort the rest of --model's list.
#
# Usage:
#   sbatch jobs/evaluate_claude.sh -c configs/eval_claude.yml
#
#   sbatch jobs/evaluate_claude.sh \
#       --model claude-opus-5 --eval-data-source bouquet_test --rpm 2000 \
#       --output results/claude_comparison.json --use-wandb --run-name claude_v1
#
# All flags forward directly to systems/claude_tokenizer/evaluate.py -- see
# `python3 evaluate.py claude_tokenizer --help`.
#
# PREREQUISITES:
#   - A real ANTHROPIC_API_KEY (export it, or pass --api-key) -- a live
#     account credential, DIFFERENT from and in addition to HF_TOKEN below.
#     count_tokens is free to call but still requires authentication and is
#     subject to your account's own RPM limit (match --rpm to your actual
#     tier -- see systems/claude_tokenizer/evaluate.py's own --rpm help).
#   - BOUQuET is ALSO a gated HF dataset (same HF_TOKEN requirement as
#     jobs/evaluate.sh/jobs/evaluate_hf_frontier.sh -- run
#     `huggingface-cli login` on this cluster, or export HF_TOKEN) --
#     needed for --eval-data-source bouquet/bouquet_test specifically, a
#     completely separate credential from ANTHROPIC_API_KEY above.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/python/3.13.3-llvm-19.1

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${ANTHROPIC_API_KEY:?No ANTHROPIC_API_KEY set -- export it before submitting (see the PREREQUISITES comment above)}"
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
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
