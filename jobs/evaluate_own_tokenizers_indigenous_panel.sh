#!/bin/bash
#SBATCH --job-name=own_tokenizers_indigenous_panel
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out common.data.indigenous_panel evaluation for all 5 of this project's
# own trained tokenizers (bpe/fanta/flexitokens/magnet/manta) in ONE job --
# see scripts/evaluate_own_tokenizers_indigenous_panel.py's own module
# docstring for why (evaluate.py's dispatcher, and jobs/evaluate.sh's own
# SLURM wrapper around it, only handle one system per invocation; this exists
# so a single cheap CPU-only eval run doesn't need five separate submissions).
# No GPU requested -- same cpu_il reasoning as jobs/evaluate.sh.
#
# Usage:
#   sbatch jobs/evaluate_own_tokenizers_indigenous_panel.sh -c configs/eval_own_tokenizers_indigenous_panel.yml
#
# All flags forward directly to scripts/evaluate_own_tokenizers_indigenous_panel.py.
#
# PREREQUISITES:
#   - `python -m common.data.prepare_indigenous_panel` must have been run once
#     already (writes to data/indigenous_panel/).
#   - configs/eval_own_tokenizers_indigenous_panel.yml's checkpoint paths must
#     point at this cluster's real checkpoints -- re-check them (`ls
#     checkpoints/`) if any of the 5 tokenizers get retrained, since a stale
#     path there fails ONLY that system (see the script's own per-system
#     error isolation -- the other systems still complete and get recorded).
#   - MAGNET CAVEAT: magnet scores 0 languages on this panel by construction
#     (see the script's own module docstring) -- not a bug, don't chase it.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: larger-quota Lustre workspace for cache/derived data -- see
# jobs/prep_pretraining_data.sh's own WORK_ROOT comment for why. Expires
# unless renewed (`ws_extend rl-tokenizers <n>`) -- only cache/derived
# data lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 2. Modules
module load devel/python/3.13.3-llvm-19.1

# 3. Environment -- HF_TOKEN only matters if a system's own checkpoint
# loading transitively touches a gated HF resource; harmless to export
# unconditionally, same convention as every other jobs/*.sh in this repo.
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
mkdir -p logs results

# 5. Run
echo "Starting own-tokenizers indigenous_panel evaluation with args: $@"
python3 -m scripts.evaluate_own_tokenizers_indigenous_panel "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
