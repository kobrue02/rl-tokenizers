#!/bin/bash
#SBATCH --job-name=pretrain_data_prep_gpu
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# GPU variant of jobs/prep_pretraining_data.sh -- for the FIVE NEURAL
# (span-family) tokenizer systems (fairtok/magnet/flexitokens/manta/fanta),
# whose induce_spans is a real torch forward pass per document (see
# pretraining/data_prep.py's own PERFORMANCE section). bpe/superbpe are pure
# Python/Rust and have NO use for a GPU here -- keep using the plain
# jobs/prep_pretraining_data.sh (cpu_il) for those two; requesting a GPU for
# them would just sit idle and waste a scarce allocation slot.
#
# The only functional difference from the CPU version: this loads the CUDA
# module (+ the same unset LD_LIBRARY_PATH cuDNN workaround every other
# GPU job in this repo uses -- see jobs/train_pretraining.sh's own comment)
# and defaults --device to cuda instead of data_prep.py's own "cpu" default.
# Running a neural tokenizer's induce_spans on CPU at real corpus scale
# (millions of documents) would be impractically slow -- this exists
# specifically to avoid that, not as a blanket "always use GPU" preference.
#
# Usage:
#   sbatch jobs/prep_pretraining_data_gpu.sh --dataset glot500 --langs all \
#       --system fanta --checkpoint checkpoints/fanta_50k.pt \
#       --vocab-json vocab_out/fanta_50k_vocab.json \
#       --output-dir pretrain_data/glot500_fanta --max-tokens 5000000000
#
# Or via a YAML config (see configs/README.md):
#   sbatch jobs/prep_pretraining_data_gpu.sh -c configs/prep_fanta_50k.yml
#
# All flags forward directly to pretraining/data_prep.py -- see
# `python3 -m pretraining.data_prep --help`.
#
# PREREQUISITE: same HF_TOKEN handling as jobs/prep_pretraining_data.sh.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # see jobs/train_pretraining.sh's own comment -- same
# cuDNN-shadowing workaround, needed here too since this also runs a
# CUDA-backed torch model (the neural tokenizer's own induce_spans).

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs pretrain_data

# 5. Run
echo "Starting GPU pretraining data prep with args: $@"
python3 -m pretraining.data_prep --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Data prep complete."
else
    echo "Data prep failed." && exit 1
fi
