#!/bin/bash
#SBATCH --job-name=pretrain_data_prep_gpu
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# GPU variant of jobs/prep/pretraining_data.sh -- for the five neural
# (span-family) systems (fairtok/magnet/flexitokens/manta/fanta), whose
# induce_spans is a real torch forward pass per document. bpe/superbpe have
# no use for a GPU here -- keep using the plain CPU script for those.
# Only functional difference: loads CUDA + defaults --device to cuda.
# AUTO-RESUBMIT: same as jobs/prep/pretraining_data.sh, --signal=B:TERM@120
# trap included -- see that script's own comment for why a plain
# post-`wait` check alone silently never fires on a genuine SLURM TIMEOUT
# (confirmed live on a real GPU prep job: 12h elapsed, zero resubmit
# message in its own log).
#
# PREREQUISITE for --dataset glot500: reads from a one-time local disk cache
# (common/data/prepare_glot500.py -- run jobs/prep/glot500.sh first), not
# live HF streaming. Pass --dataset-config (or a config's dataset_config
# field) pointing at the cache if it's not at the default GLOT500_LOCAL_DIR.
#
# Usage:
#   sbatch jobs/prep/pretraining_data_gpu.sh --dataset glot500 --langs all \
#       --dataset-config "$WORK_ROOT/data/glot500" \
#       --system fanta --checkpoint checkpoints/fanta_50k.pt \
#       --vocab-json vocab_out/fanta_50k_vocab.json \
#       --output-dir pretrain_data/glot500_fanta --max-tokens 5000000000
#   sbatch jobs/prep/pretraining_data_gpu.sh -c configs/prep/fanta_50k.yml
#
# All flags forward directly to systems/pretraining/data_prep.py -- see
# `python3 -m systems.pretraining.data_prep --help`. Requires HF_TOKEN.

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
mkdir -p logs pretrain_data

OUTPUT_DIR=$(python3 -c "
import sys
from systems.pretraining.data_prep import build_arg_parser
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

# See jobs/prep/pretraining_data.sh's own comment for the full rationale --
# shared between the normal post-`wait` path and the SIGTERM trap below.
check_and_resubmit() {
    if [ -f "$OUTPUT_DIR/shards_meta.json" ]; then
        echo "Data prep complete."
        exit 0
    fi

    echo "shards_meta.json not found in $OUTPUT_DIR -- checking for progress to resume from."
    AFTER_TOKENS=$(checkpoint_tokens "$CKPT_PATH")
    if [ "$AFTER_TOKENS" -le "$BEFORE_TOKENS" ]; then
        echo "No progress made this run (before=$BEFORE_TOKENS after=$AFTER_TOKENS tokens) -- NOT resubmitting. Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err." >&2
        exit 1
    fi

    TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'TimeLimit=\K\S+')

    echo "Progress made this run: $BEFORE_TOKENS -> $AFTER_TOKENS tokens. Resubmitting..."
    sbatch --time="$TIME_LIMIT" jobs/prep/pretraining_data_gpu.sh "$@"
    SBATCH_EXIT=$?
    if [ "$SBATCH_EXIT" -ne 0 ]; then
        echo "Resubmission via sbatch failed (exit $SBATCH_EXIT) -- resume manually with:" >&2
        echo "  sbatch --time=$TIME_LIMIT jobs/prep/pretraining_data_gpu.sh $@" >&2
        exit 1
    fi
    echo "Resubmitted successfully."
    exit 0
}

on_term() {
    echo "Received SIGTERM (120s from --signal=B:TERM@120) -- stopping the tokenizer and resubmitting early."
    kill -TERM "$CHILD_PID" 2>/dev/null
    wait "$CHILD_PID" 2>/dev/null
    check_and_resubmit
}
trap on_term TERM

echo "Starting GPU pretraining data prep with args: $@"
python3 -m systems.pretraining.data_prep --device cuda "$@" &
CHILD_PID=$!
wait "$CHILD_PID"
check_and_resubmit
