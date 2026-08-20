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

# Downstream benchmark eval (XNLI/XCOPA/FLORES-MT) for a
# systems.pretraining.train checkpoint. Single GPU, no DDP -- run multiple
# checkpoints/benchmarks as separate submissions.
# GPU is reserved uniformly since FLORES-MT's generate() has no KV cache
# (recomputes the full prefix every step -- see model.py) and would be slow
# on CPU past the tiny/small presets, even though XNLI/XCOPA don't need it.
# --time=04:00:00 is a starting estimate -- widen for flores_mt with a large
# --max-examples on a bigger preset (generate()'s per-token cost dominates).
#
# Usage:
#   sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark xnli --langs en,de,fr,ar,zh --max-examples 1000 \
#       --output results/xnli_bpe.json
#   sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --benchmark xnli,xcopa,flores_mt --langs en,de,fr --lang-pairs eng:spa,eng:arz \
#       --max-examples 500 --output results/all_bpe.json --use-wandb --run-name eval_bpe_50k
#   # --benchmark takes a comma-separated list -- one combined results file;
#   # --langs only applies to xnli/xcopa, --lang-pairs only to flores_mt.
#
# All flags forward directly -- see `python3 -m systems.pretraining.cli_eval --help`.

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

echo "Starting pretraining evaluation with args: $@"
python3 -m systems.pretraining.cli_eval --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation complete."
else
    echo "Evaluation failed." && exit 1
fi
