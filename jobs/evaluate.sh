#!/bin/bash
#SBATCH --job-name=tokenizer_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation for any trained checkpoint (fairtok/magnet/
# flexitokens/manta/etc, see evaluate.py's dispatcher). CPU-only.
# --time=08:00:00: the full test split (259 languages, ~272k rows) via
# evaluate_on_groups's unbatched per-sequence loop confirmed to time out at
# the original 30min budget on a real FANTA run -- if 8h isn't enough
# either, batch evaluate_on_groups itself rather than widening this further.
#
# Usage: sbatch jobs/evaluate.sh fairtok --checkpoint checkpoints/policy_12345.pt
#   sbatch jobs/evaluate.sh magnet --checkpoint checkpoints/magnet_12345.pt --num-groups 50
# First arg is the tokenizer name; remaining flags forward to that
# tokenizer's own evaluate.py -- see `python evaluate.py <tokenizer> --help`.
# Requires HF_TOKEN (bouquet is gated).

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

TOKENIZER=$1
shift
echo "Evaluating $TOKENIZER with args: $@"
python3 evaluate.py "$TOKENIZER" "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
