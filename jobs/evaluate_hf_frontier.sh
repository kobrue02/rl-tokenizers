#!/bin/bash
#SBATCH --job-name=hf_frontier_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation for one or more ARBITRARY HuggingFace models'
# own tokenizers -- see systems/hf_frontier/evaluate.py's own module
# docstring for the multi-repo/--output design and systems/hf_frontier/
# model.py for how tokenizer-only loading + byte-span reconstruction work.
# No GPU requested: this only ever tokenizes text (no model inference at
# all -- see model.py's own docstring confirming NO model weights are ever
# downloaded), same cpu_il reasoning as jobs/evaluate.sh.
#
# --time=04:00:00: scoring N frontier tokenizers against the full BOUQuET
# test split (~272k rows) is N times jobs/evaluate.sh's own workload for a
# single from-scratch checkpoint (that job needed up to 8h for ONE
# tokenizer, see its own comment) -- widen this if you list more than a
# couple of --hf-repo-id entries, or narrow the run with --num-groups /
# --eval-data-source bouquet (dev, not test) for a quicker exploratory pass.
#
# Usage:
#   sbatch jobs/evaluate_hf_frontier.sh -c configs/eval_hf_frontier.yml
#
#   sbatch jobs/evaluate_hf_frontier.sh \
#       --hf-repo-id deepseek-ai/DeepSeek-V4-Pro,moonshotai/Kimi-K3,meta-llama/Llama-3.1-8B-Instruct \
#       --trust-remote-code --eval-data-source bouquet_test \
#       --output results/hf_frontier_comparison.json --use-wandb --run-name frontier_v1
#
# All flags forward directly to systems/hf_frontier/evaluate.py -- see
# `python3 evaluate.py hf_frontier --help`.
#
# PREREQUISITES:
#   - BOUQuET is a gated HF dataset (same HF_TOKEN requirement as
#     jobs/evaluate.sh -- run `huggingface-cli login` on this cluster, or
#     export HF_TOKEN, before submitting).
#   - --trust-remote-code executes that repo's OWN Python code (needed for
#     e.g. moonshotai/Kimi-K3) -- only pass it if you've reviewed what
#     that implies, see systems/hf_frontier/model.py's own docstring.
#   - meta-llama/Llama-3.1-8B-Instruct (and other gated model repos) need
#     their license accepted on huggingface.co AND your HF_TOKEN to
#     actually have that access granted -- a plain HF_TOKEN without
#     accepted access will fail to load that specific repo's tokenizer,
#     even though BOUQuET's own access is fine.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/python/3.13.3-llvm-19.1

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN and no cached login at ~/.cache/huggingface/token -- run \`huggingface-cli login\` on this cluster (not just your laptop) or export HF_TOKEN before submitting}"
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

# 5. Run
echo "Starting hf_frontier evaluation with args: $@"
python3 evaluate.py hf_frontier "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
