#!/bin/bash
#SBATCH --job-name=pretrain_data_prep
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Offline tokenization: stream a corpus from common/corpora.py's shared
# registry -- the SAME registry common/cli_data.py's tokenizer-training side
# uses (oldi_seed/flores_dev/smol/glot500/fineweb_edu/olmo_mix, no separate
# pretraining-only source list) -- tokenize with an already-trained systems/
# checkpoint, pack into token shards. See pretraining/data_prep.py's own
# module docstring for the full design and its PERFORMANCE section for a
# real, stated caveat: the five neural (span-family) tokenizer systems
# encode one document at a time, no batching, and fairtok specifically
# loops one byte at a time in Python -- expect this to be substantially
# slower than bpe/superbpe for a genuinely large prep run.
#
# No GPU requested: this is CPU-bound (network streaming + per-document
# tokenization), matching jobs/evaluate.sh's own cpu_il choice for the same
# reason. --time=12:00:00 is a starting estimate, not a benchmark at real
# scale (only tested locally against a single small Glot500 language) --
# widen it if a --langs/--dataset-config/--max-tokens combination needs longer.
#
# Usage:
#   sbatch jobs/prep_pretraining_data.sh --dataset glot500 --langs all \
#       --system bpe --checkpoint checkpoints/bpe_12345.json \
#       --output-dir pretrain_data/glot500_bpe --max-tokens 5000000000
#
#   sbatch jobs/prep_pretraining_data.sh --dataset fineweb_edu --dataset-config sample-10BT \
#       --system superbpe --checkpoint checkpoints/superbpe_12345.pt \
#       --output-dir pretrain_data/fineweb_superbpe
#
#   sbatch jobs/prep_pretraining_data.sh --dataset glot500 --langs all \
#       --system fanta --checkpoint checkpoints/fanta_12345.pt --vocab-json vocab_out/fanta_vocab_12345.json \
#       --output-dir pretrain_data/glot500_fanta --max-tokens 500000000
#
#   sbatch jobs/prep_pretraining_data.sh --dataset oldi_seed --langs all \
#       --system fanta --checkpoint checkpoints/fanta_12345.pt --vocab-json vocab_out/fanta_vocab_12345.json \
#       --output-dir pretrain_data/oldi_fanta
#   # ^ a genuinely parallel source used for pretraining instead of the usual
#   # monolingual choices -- small relative to fineweb_edu/olmo_mix/glot500,
#   # but exactly as available now that both halves of this project share one
#   # registry: no source is off-limits to either consumer.
#
# All flags forward directly to pretraining/data_prep.py -- see
# `python3 -m pretraining.data_prep --help`.
#
# PREREQUISITE: fineweb-edu/olmo-mix/glot500 are all public but sizeable HF
# datasets -- same HF_TOKEN handling as jobs/train.sh (a token isn't
# strictly required for these three specifically, unlike flores_plus/
# bouquet, but is set anyway since a logged-in session avoids anonymous
# rate limits on a long streaming run).

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 2. Modules
module load devel/python/3.13.3-llvm-19.1

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
echo "Starting pretraining data prep with args: $@"
python3 -m pretraining.data_prep "$@"

if [ $? -eq 0 ]; then
    echo "Data prep complete."
else
    echo "Data prep failed." && exit 1
fi
