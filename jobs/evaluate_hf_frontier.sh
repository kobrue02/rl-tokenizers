#!/bin/bash
#SBATCH --job-name=hf_frontier_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation for arbitrary HF models' own tokenizers
# (systems/tokenization/hf_frontier/evaluate.py) -- tokenizer-only, no model
# weights downloaded, no GPU needed. One repo failing doesn't abort the rest.
# --time=11:00:00: not benchmarked at the current 33-repo scale; widen if you add more repos.
#
# Usage:
#   sbatch jobs/evaluate_hf_frontier.sh -c configs/eval_hf_frontier.yml
#   sbatch jobs/evaluate_hf_frontier.sh \
#       --hf-repo-id deepseek-ai/DeepSeek-V4-Pro,moonshotai/Kimi-K3,meta-llama/Llama-3.1-8B-Instruct \
#       --trust-remote-code --eval-data-source bouquet_test \
#       --output results/hf_frontier_comparison.json --use-wandb --run-name frontier_v1
# All flags forward directly -- see `python3 evaluate.py hf_frontier --help`.
#
# PREREQUISITES: HF_TOKEN (BOUQuET is gated). --trust-remote-code executes
# that repo's own Python code (needed for e.g. moonshotai/Kimi-K3) -- only
# pass it if you've reviewed what that implies. Gated repos in the current
# 33-repo list needing accepted access: meta-llama/Llama-3.1-8B-Instruct,
# meta-llama/Llama-3.3-70B-Instruct, google/gemma-7b. The 7 "tiktoken:{name}"
# entries aren't HF Hub repos -- loaded via the `tiktoken` package directly.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

module load devel/python/3.13.3-llvm-19.1

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN -- run \`huggingface-cli login\` or export HF_TOKEN}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

echo "Starting hf_frontier evaluation with args: $@"
python3 evaluate.py hf_frontier "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
