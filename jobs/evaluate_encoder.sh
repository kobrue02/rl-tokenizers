#!/bin/bash
#SBATCH --job-name=encoder_eval
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

# Glot500-style eval (pseudoperplexity / sentence retrieval / roundtrip
# alignment -- systems/pretraining/encoder_eval.py) for an
# encoder_train.py checkpoint. Single GPU, no DDP -- run multiple
# checkpoints/benchmarks as separate submissions. Mirrors
# jobs/evaluate_pretrained.sh's own structure for the decoder's eval suite.
# GPU is reserved uniformly since pseudoperplexity does one forward pass
# PER TOKEN per sentence (see encoder_eval.pseudo_perplexity's own
# docstring) -- can dominate wall-clock on a large eval set even though
# retrieval/roundtrip's own forward passes are cheap by comparison.
# --time=04:00:00 is a starting estimate -- widen for --benchmark pppl over
# a large corpus.
#
# --benchmark roundtrip and any --dataset bible_nlp run need bible_nlp
# prepared locally FIRST (see jobs/prepare_bible_nlp.sh) -- common.data.
# corpora.stream_groups("bible_nlp", ...) has no live-streaming fallback.
#
# Usage:
#   sbatch jobs/evaluate_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark retrieval --dataset tatoeba_mt --pair deu-eng --split test
#   sbatch jobs/evaluate_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark roundtrip --cycle-langs eng,fra,deu,eng
#   sbatch jobs/evaluate_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark pppl --dataset tatoeba_mt --pair deu-eng --split test --lang deu
#
# All flags forward directly -- see `python3 -m systems.pretraining.encoder_cli_eval --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

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

echo "Starting encoder evaluation with args: $@"
python3 -m systems.pretraining.encoder_cli_eval --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
