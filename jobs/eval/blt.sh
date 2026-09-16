#!/bin/bash
#SBATCH --job-name=blt_eval
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Held-out BOUQuET evaluation of BLT's entropy-based dynamic patch boundaries
# (systems/tokenization/blt/evaluate.py). UNLIKE hf_frontier's eval, this runs
# a real ~100M-param transformer forward pass per sentence, not free
# tokenization -- no GPU needed (CPU-only, confirmed working), but genuinely
# slow: measured live on this project's own dev machine (4 CPU threads,
# unbatched -- batching gave ~1.0x speedup, i.e. none, since this workload is
# compute-bound not overhead-bound) at ~54s per BOUQuET group (up to 259
# languages/group, real sentence/paragraph lengths averaging ~157 bytes, up to
# ~2000 for paragraph-level entries -- NOT the ~30ms/sentence a short canary
# string would suggest). Extrapolated over bouquet_test's 1052 groups: ~16
# HOURS. --time=20:00:00 is padding on top of that extrapolation (measured
# from only 3 groups) -- widen further if this job runs out of time before
# finishing, and consider re-benchmarking with a larger --num-groups sample
# first if you add --cpus-per-task beyond 8.
#
# Usage:
#   sbatch jobs/eval/blt.sh -c configs/eval/blt.yml
#   sbatch jobs/eval/blt.sh --eval-data-source bouquet_test \
#       --output results/blt_comparison.json --use-wandb --run-name blt_v1
#   # quick sanity check on a small sample first (a few minutes, not hours):
#   sbatch jobs/eval/blt.sh --eval-data-source bouquet_test --num-groups 20 \
#       --output results/blt_comparison_sample20.json
# All flags forward directly -- see `python3 evaluate.py blt --help`.
#
# PREREQUISITES: HF_TOKEN with GATED ACCESS APPROVED for facebook/blt-1b
# (huggingface.co/facebook/blt-1b -- request access, wait for approval; this
# is a personal license acceptance, not something scriptable). Loads
# facebook/blt-1b's own copy of entropy_model/consolidated.pth -- NOT the
# separately-gated facebook/blt-entropy repo (see systems/tokenization/blt/
# model.py's own comment on _ENTROPY_MODEL_REPO for why).

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

echo "Starting blt evaluation with args: $@"
python3 evaluate.py blt "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
