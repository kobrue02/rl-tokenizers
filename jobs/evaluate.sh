#!/bin/bash
#SBATCH --job-name=tokenizer_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation for any trained checkpoint in this repo (fairtok/
# magnet/flexitokens/manta) -- see evaluate.py's module docstring for the dispatch
# pattern and common/eval_common.py for the shared scoring logic every tokenizer's
# own evaluate.py uses. This is a handful of forward passes over BOUQuET dev, not a
# training loop, so no GPU is requested -- cpu_il is this cluster's CPU-only
# counterpart to jobs/train*.sh's gpu_a100_il partition (same "_il" allocation,
# confirmed via `sinfo`). evaluate.py's own --device flag defaults to "cpu"
# already, matching this partition.
#
# Usage:
#   sbatch jobs/evaluate.sh fairtok --checkpoint checkpoints/policy_12345.pt
#   sbatch jobs/evaluate.sh magnet --checkpoint checkpoints/magnet_12345.pt --num-groups 50
#   sbatch jobs/evaluate.sh flexitokens --checkpoint checkpoints/flexitokens_12345.pt
#   sbatch jobs/evaluate.sh manta --checkpoint checkpoints/manta_12345.pt
#
# First positional arg is the tokenizer name (forwarded to evaluate.py's own
# dispatcher, same names train.py uses); every flag after that is forwarded
# directly to THAT tokenizer's own evaluate.py -- see
# `python evaluate.py <tokenizer> --help`. --checkpoint is required there (no
# default path is guessed here).
#
# PREREQUISITE: bouquet is a gated HF dataset (see common/oldi_data.py). Same
# HF_TOKEN handling as jobs/train.sh -- run `huggingface-cli login` on this cluster
# (not just your laptop), or export HF_TOKEN before submitting.

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

# 5. Run
TOKENIZER=$1
shift
echo "Evaluating $TOKENIZER with args: $@"
python3 evaluate.py "$TOKENIZER" "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
