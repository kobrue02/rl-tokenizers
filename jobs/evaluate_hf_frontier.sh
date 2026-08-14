#!/bin/bash
#SBATCH --job-name=hf_frontier_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
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
# --time=08:00:00: a CONSERVATIVE guess, not a benchmarked number at this
# scale (configs/eval_hf_frontier.yml currently lists 16 repos) -- unlike
# jobs/evaluate.sh's own from-scratch checkpoints, a frontier tokenizer's
# encode() is plain BPE/SentencePiece merge application with no neural net
# at all, so per-row tokenization cost over BOUQuET test's ~272k rows should
# be genuinely fast; the real per-repo cost is mostly download/load overhead
# (fetching each repo's own vocab/merges files, then building the tokenizer
# object) repeated 16 times, not compute that scales with row count the way
# jobs/evaluate.sh's own neural forward-pass cost does. Widen this if you add
# even more repos, or narrow a run with --num-groups / --eval-data-source
# bouquet (dev, not test) for a quicker exploratory pass. One repo failing
# (see systems/hf_frontier/evaluate.py's own per-repo error isolation)
# doesn't abort the rest of the list, so a bad/gated-without-access repo in
# the list costs time on just that one repo, not the whole job.
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
#   - Gated repos (need their license accepted on huggingface.co AND your
#     HF_TOKEN to actually have that access granted -- a plain HF_TOKEN
#     without accepted access fails to load just that one repo's tokenizer,
#     even though BOUQuET's own access is fine, and now doesn't abort the
#     rest of the list either): in configs/eval_hf_frontier.yml's current
#     16-repo list, that's meta-llama/Llama-3.1-8B-Instruct, meta-llama/
#     Llama-3.3-70B-Instruct, and google/gemma-7b.

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
