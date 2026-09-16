#!/bin/bash
#SBATCH --job-name=morphology_all
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Computes MED/Consistency-F1 (common.eval.morphology) for every tokenizer
# EXCEPT claude_tokenizer, in ONE job (scripts/evaluate_morphology_all.py)
# -- evaluate.py's dispatcher only handles one system per invocation.
# --cpus-per-task=16: same as jobs/eval/own_tokenizers_indigenous_panel.sh
# (a real run there OOM-killed at 4 cores partway through a scoring loop) --
# carried forward as a generous default, not independently benchmarked here.
#
# Usage: sbatch jobs/eval/morphology_all.sh -c configs/eval/morphology_all.yml
#
# PREREQUISITES: `python3 -m common.data.prepare_morphology_gold
# --output-dir data/morphology_gold` run once already. Config's checkpoint
# paths must point at real checkpoints -- a stale path fails only that one
# system, others still complete. Requires HF_TOKEN (bouquet is gated, and
# hf_frontier's gated repos need license acceptance -- see
# configs/eval/hf_frontier.yml's own comment).

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep/pretraining_data.sh

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

echo "Starting morphology_all evaluation with args: $@"
python3 -m scripts.evaluate_morphology_all "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
