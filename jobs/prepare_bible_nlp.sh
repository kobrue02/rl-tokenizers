#!/bin/bash
#SBATCH --job-name=prepare_bible_nlp
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# One-time local prep for bible-nlp/biblenlp-corpus (common/data/prepare_bible_nlp.py)
# -- run ONCE before --data-source bible_nlp is usable (no live fallback).
# NOT resumable -- a killed run's own metadata.json is only written at the
# end, so rerun from scratch, not a --limit subset, if it times out.
# CPU-only. Usage:
#   sbatch jobs/prepare_bible_nlp.sh --output-dir data/bible_nlp
#   sbatch jobs/prepare_bible_nlp.sh --output-dir data/bible_nlp_test --limit 20   # quick sanity check
# All flags forward directly -- see `python3 -m common.data.prepare_bible_nlp --help`.

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
mkdir -p logs data

echo "Starting bible_nlp prep with args: $@"
python3 -m common.data.prepare_bible_nlp "$@"

if [ $? -eq 0 ]; then
    echo "bible_nlp prep complete."
else
    echo "bible_nlp prep failed." && exit 1
fi
