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
# AUTO-RESUBMIT: mirrors jobs/train_pretraining.sh's own and
# jobs/prep_pretraining_data.sh's own (see that script's own comment for the
# full rationale) -- a run whose --max-tokens exceeds this job's own --time
# limit gets killed mid-run; pretraining.data_prep.prep_dataset checkpoints
# its own progress periodically and resumes automatically (no extra flag)
# whenever rerun against the same --output-dir (see its own RESUME
# docstring section), so this script resubmits itself when progress was
# made but the run didn't finish, and refuses to when it wasn't (avoiding
# an infinite resubmission loop on a real, persistent failure).
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

# 5. Resolve this run's output_dir from the EXACT args this job received --
# reuses pretraining.data_prep's own parsing so this can never drift from
# what it actually uses.
OUTPUT_DIR=$(python3 -c "
import sys
from pretraining.data_prep import build_arg_parser
from common.config_file import parse_args_with_config
args = parse_args_with_config(build_arg_parser(), sys.argv[1:])
print(args.output_dir)
" "$@")
CKPT_PATH="$OUTPUT_DIR/prep_checkpoint.json"

checkpoint_tokens() {
    python3 -c "
import json
try:
    with open('$1') as f:
        print(json.load(f).get('total_tokens', 0))
except FileNotFoundError:
    print(0)
"
}
BEFORE_TOKENS=$(checkpoint_tokens "$CKPT_PATH")

# 6. Run
echo "Starting GPU pretraining data prep with args: $@"
python3 -m pretraining.data_prep --device cuda "$@"
PREP_EXIT=$?

# 7. Done, or resubmit? See the AUTO-RESUBMIT comment at the top.
if [ -f "$OUTPUT_DIR/shards_meta.json" ]; then
    echo "Data prep complete."
    exit 0
fi

echo "shards_meta.json not found in $OUTPUT_DIR (exited with code $PREP_EXIT) -- checking for progress to resume from."
AFTER_TOKENS=$(checkpoint_tokens "$CKPT_PATH")
if [ "$AFTER_TOKENS" -le "$BEFORE_TOKENS" ]; then
    echo "No progress made this run (before=$BEFORE_TOKENS after=$AFTER_TOKENS tokens) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
    exit 1
fi

# This job's OWN actual --time limit (may have been overridden at
# submission) -- a bare resubmit would otherwise silently fall back to this
# script's #SBATCH --time=12:00:00 default. Same reasoning as
# jobs/prep_pretraining_data.sh's own TIME_LIMIT/--gres preservation.
TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

echo "Progress made this run: $BEFORE_TOKENS -> $AFTER_TOKENS tokens. Resubmitting..."
sbatch --time="$TIME_LIMIT" jobs/prep_pretraining_data_gpu.sh "$@"
SBATCH_EXIT=$?
if [ "$SBATCH_EXIT" -ne 0 ]; then
    echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
    echo "  sbatch --time=$TIME_LIMIT jobs/prep_pretraining_data_gpu.sh $@" >&2
    exit 1
fi
echo "Resubmitted successfully."
exit 0
