#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# EleutherAI/lm-evaluation-harness eval (xnli/xcopa/blimp/xstorycloze/
# lambada_multilingual/global_piqa -- see systems/pretraining/
# lm_eval_adapter.py's own module docstring for the full list and why each
# one) for a systems.pretraining.train checkpoint, via cli_lm_eval.py. The
# "use the standard harness" counterpart to jobs/eval/pretrained.sh's
# own hand-rolled xnli/xcopa/flores_mt eval -- same single-GPU, no-DDP
# shape, same env setup, just a different Python entry point.
#
# No AUTO-RESUBMIT (unlike jobs/prep/pretraining_data.sh/jobs/pretrain/pretraining.sh):
# cli_lm_eval.py's own lm_eval.simple_evaluate() call has no incremental
# checkpointing at all -- a run killed by the time limit has NOTHING to
# resume from, a resubmit would just restart the whole thing from scratch
# regardless, so there's no point pretending otherwise with a signal trap.
# Size --time (and/or use --limit, see below) so the run actually finishes
# inside one job instead.
#
# UNBATCHED (see lm_eval_adapter.py's own docstring): this scores ONE
# example at a time, no throughput batching -- --tasks spanning many
# languages/paradigms (e.g. the full blimp's 67 paradigms, or global_piqa's
# ~130+ language/dialect configs) with no --limit cap can genuinely take
# hours. Start with a SMALL --limit (e.g. 200) to sanity-check the run and
# get a real per-example timing estimate before committing to a full,
# uncapped --time=04:00:00-or-more run.
#
# Usage:
#   sbatch jobs/eval/lm_eval.sh -c configs/eval/lm_eval_bpe_culturax.yml
#   sbatch jobs/eval/lm_eval.sh -c configs/eval/lm_eval_bpe_culturax.yml --limit 200
#   # -c config.yml resolves checkpoint/system/tokenizer-checkpoint/tasks/
#   # etc. for you -- see configs/eval/lm_eval_bpe_culturax.yml for the field
#   # names (identical to `python3 -m systems.pretraining.cli_lm_eval --help`'s
#   # own flags, underscored). A flag passed explicitly on the command line
#   # (like the --limit override above) wins over the same key in the YAML
#   # file, same precedence rule every config file in this project follows
#   # (see configs/README.md).
#   sbatch --partition=gpu_h100 jobs/eval/lm_eval.sh -c configs/eval/lm_eval_bpe_culturax.yml
#   # gpu_h100 is bwUniCluster 3.0's dedicated H100 queue -- see
#   # jobs/pretrain/pretraining.sh's own comment for why it's often the better
#   # choice over the shared gpu_a100_il/gpu_h100_il pool this script
#   # defaults to.
#
# All flags forward directly -- see `python3 -m systems.pretraining.cli_lm_eval --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep/pretraining_data.sh

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # avoids the cuda module's older cuDNN shadowing PyTorch's bundled one

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

echo "Starting lm-evaluation-harness run with args: $@"
python3 -m systems.pretraining.cli_lm_eval --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
