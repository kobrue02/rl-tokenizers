#!/bin/bash
#SBATCH --job-name=own_tokenizers_indigenous_panel
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out indigenous_panel evaluation for all of this project's own trained
# tokenizers in ONE job (scripts/evaluate_own_tokenizers_indigenous_panel.py)
# -- evaluate.py's dispatcher only handles one system per invocation.
# --cpus-per-task=16: CONFIRMED LIVE a run OOM-killed at 4 cores/7.81GB
# (cluster ties memory to cpus-per-task at ~1.95GB/core) partway into
# fanta's scoring loop over the full combined panel -- 16 cores is a
# generous widening, not benchmarked; profile further if it OOMs again.
#
# Usage: sbatch jobs/evaluate_own_tokenizers_indigenous_panel.sh -c configs/eval_own_tokenizers_indigenous_panel.yml
#
# PREREQUISITES: `python -m common.data.prepare_indigenous_panel` run once
# already. Config's checkpoint paths must point at real checkpoints -- a
# stale path fails only that one system, others still complete. MAGNET
# scores 0 languages on this panel by construction -- not a bug.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

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
mkdir -p logs results

echo "Starting own-tokenizers indigenous_panel evaluation with args: $@"
python3 -m scripts.evaluate_own_tokenizers_indigenous_panel "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
