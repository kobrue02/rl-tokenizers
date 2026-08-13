#!/bin/bash
#SBATCH --job-name=pretrain_generate
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Generates qualitative text samples from a pretraining.train checkpoint --
# see pretraining/cli_generate.py's own docstring. Chain this via
# --dependency=afterok after jobs/train_pretraining.sh so a fixed prompt
# panel's completions get generated and saved automatically once training
# finishes, without a manual step someone has to remember to run (ad-hoc,
# interactive generation with an arbitrary one-off prompt is still supported
# directly via `python3 -m pretraining.cli_generate`, no sbatch needed for
# that -- this script is specifically for making sample generation a durable,
# automatic part of the pipeline).
#
# --time=00:30:00: a handful of prompts x a handful of samples each, at
# model.generate()'s own unbatched/no-KV-cache cost (see model.py) -- fast
# for tiny/small/medium, widen this if pointed at a much larger preset with
# many --prompt/--num-samples combinations.
#
# Usage (chained after a training job with id $train_id):
#   sbatch --dependency=afterok:$train_id jobs/generate_samples.sh \
#       --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
#       --prompt "The quick brown fox" --prompt "Once upon a time" \
#       --prompt "The history of the Roman Empire" \
#       --max-new-tokens 100 --num-samples 3 --output results/samples_bpe.json \
#       --use-wandb --wandb-project pretraining --run-name generate_bpe_50k
#   # ^ --use-wandb logs every generated sample as a wandb.Table (job_type=
#   # "generate"), in the SAME project pretraining.train/cli_eval use --
#   # browse actual generated text in the wandb UI instead of only results/*.json.
#
# All flags forward directly to pretraining/cli_generate.py -- see
# `python3 -m pretraining.cli_generate --help`.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # see jobs/train_pretraining.sh's own comment -- same
# cuDNN-shadowing workaround, needed here too since this also runs a
# CUDA-backed torch model.

# 3. Environment
export PYTHONUNBUFFERED=1

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

# 5. Run
echo "Starting sample generation with args: $@"
python3 -m pretraining.cli_generate --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Sample generation complete."
else
    echo "Sample generation failed." && exit 1
fi
