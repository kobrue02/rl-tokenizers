#!/bin/bash
#SBATCH --job-name=pretrain_eval
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

# Downstream benchmark evaluation (XNLI/XCOPA/FLORES-MT) for a
# pretraining.train checkpoint -- see pretraining/benchmarks.py and
# pretraining/eval_harness.py for the loaders/scorers this calls into.
# Single process, no torchrun: unlike jobs/train_pretraining.sh, evaluation
# here is one model, one GPU, no DDP -- multiple benchmarks/checkpoints are
# just multiple separate `sbatch` submissions of this same script.
#
# --gres=gpu:1: XNLI/XCOPA scoring (evaluate_multiple_choice) is just forward
# passes and would run fine on CPU even for the larger presets, but FLORES-MT
# (evaluate_translation) calls TransformerLM.generate, which has no KV cache
# (see model.py's own docstring) and recomputes the full prefix every step --
# slow enough on CPU at anything past the "tiny"/"small" presets that a GPU
# is worth reserving here uniformly, even though it isn't a hard requirement
# for xnli/xcopa specifically.
#
# --time=04:00:00 is a starting estimate (this has only been run against the
# infrastructure's own smoke test, not a real checkpoint at real benchmark
# scale) -- widen it for --benchmark flores_mt with a large --max-examples on
# a larger model preset, since generate()'s per-token cost there is the
# actual bottleneck, not the model's forward-pass cost alone.
#
# Usage:
#   sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark xnli --langs en,de,fr,ar,zh --max-examples 1000 \
#       --output results/xnli_bpe.json
#
#   sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
#       --system fanta --tokenizer-checkpoint checkpoints/fanta_12345.pt \
#       --vocab-json vocab_out/fanta_vocab_12345.json \
#       --benchmark flores_mt --lang-pairs eng:spa,eng:arz,eng:ben \
#       --max-examples 200 --output results/flores_fanta.json
#
#   sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark xnli,xcopa,flores_mt --langs en,de,fr --lang-pairs eng:spa,eng:arz \
#       --max-examples 500 --output results/all_bpe.json \
#       --use-wandb --wandb-project pretraining --run-name eval_bpe_50k
#   # ^ --benchmark takes a comma-separated list -- one job, one combined
#   # results file (keyed by benchmark name) instead of one sbatch call per
#   # benchmark; --langs only applies to xnli/xcopa, --lang-pairs only to
#   # flores_mt, each benchmark in the list ignores the flag it has no use for.
#   # --use-wandb logs every metric above (plus flores_mt's raw generated
#   # samples as a wandb.Table) as job_type="eval", in the SAME project
#   # pretraining.train's own run used (job_type="train") -- filter by
#   # job_type in the wandb UI to see just one stage, or by run name to line
#   # up a specific eval against the training run it evaluated.
#
# All flags forward directly to pretraining/cli_eval.py -- see
# `python3 -m pretraining.cli_eval --help`.

# 1. Project root -- UPDATE THIS to wherever this repo actually lives on the cluster
PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# WORK_ROOT: larger-quota Lustre workspace for cache/derived data -- see
# jobs/prep_pretraining_data.sh's own WORK_ROOT comment for why. Expires
# unless renewed (`ws_extend rl-tokenizers <n>`) -- only cache/derived
# data lives here, never code.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

# 2. Modules
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # see jobs/train_pretraining.sh's own comment -- same
# cuDNN-shadowing workaround, needed here too since this also runs a
# CUDA-backed torch model.

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results

# 5. Run
echo "Starting pretraining evaluation with args: $@"
python3 -m pretraining.cli_eval --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
