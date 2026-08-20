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

# Generates qualitative text samples from a systems.pretraining.train
# checkpoint (systems/pretraining/cli_generate.py). Chain via
# --dependency=afterok after jobs/train_pretraining.sh to auto-generate a
# fixed prompt panel once training finishes (ad-hoc generation with an
# arbitrary prompt is also directly runnable without sbatch).
#
# Usage (chained after a training job with id $train_id):
#   sbatch --dependency=afterok:$train_id jobs/generate_samples.sh \
#       --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
#       --prompt "The quick brown fox" --prompt "Once upon a time" \
#       --max-new-tokens 100 --num-samples 3 --output results/samples_bpe.json \
#       --use-wandb --wandb-project pretraining --run-name generate_bpe_50k
# All flags forward directly -- see `python3 -m systems.pretraining.cli_generate --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # avoids the cuda module's older cuDNN shadowing PyTorch's bundled one

export PYTHONUNBUFFERED=1

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

echo "Starting sample generation with args: $@"
python3 -m systems.pretraining.cli_generate --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Sample generation complete."
else
    echo "Sample generation failed." && exit 1
fi
