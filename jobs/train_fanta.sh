#!/bin/bash
#SBATCH --job-name=fanta_train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# FANTA training -- MANTa's architecture (fanta/model.py re-exports MantaModel
# unchanged), trained with next-byte cross-entropy PLUS two added terms: a
# differentiable Gini-coefficient penalty over each language's mean compression
# rate within a batch (fairness_loss/differentiable_gini), and a per-language rate
# ANCHOR (rate_anchor_loss) pulling each language toward its own target rate --
# see fanta/model.py's module docstring for why the anchor exists (the Gini term
# alone has a degenerate "equally uncompressed" solution, confirmed empirically on
# a real run, not just anticipated). Unlike jobs/train_manta.sh, batching here is
# GROUP-based, not flat individual-sentence sampling (see fanta/train.py's module
# docstring for why: both loss terms need several languages' compression rates in
# the SAME forward pass). Has the same --use-wandb/--wandb-project/--run-name
# flags every other job here does (see fanta/train.py's FantaConfig).
#
# Usage:
#   sbatch jobs/train_fanta.sh --data-source all --langs all --max-steps 20000 --vocab-size 50000
#   sbatch jobs/train_fanta.sh --data-source oldi_seed --max-steps 2000   # quicker, single source
#   sbatch jobs/train_fanta.sh --lambda-fair 5.0 --lambda-rate 3.0 ...   # reweight either term
#   sbatch jobs/train_fanta.sh --target-rate-anchor 6.0 --anchor-lang eng ...   # change the rate target
#
# All train.py fanta / fanta.cli flags are forwarded directly -- see
# `python train.py fanta --help`.
#
# PREREQUISITE: flores_plus and bouquet are gated HF datasets (see common/data/oldi_data.py,
# reused here for data loading). Same HF_TOKEN handling as jobs/train.sh -- see below.

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
# This module puts its OWN (older) cuDNN on LD_LIBRARY_PATH, which shadows the
# newer cuDNN PyTorch already bundles for itself (the nvidia-cudnn-cu12 wheel in
# .venv) -- FANTA reuses MantaModel's block-level GRU unchanged, so it hits the
# exact same "PyTorch was compiled against (9, 20, 0) but found runtime version
# (9, 7, 1)" issue jobs/train_manta.sh does the moment it runs on a CUDA tensor.
# PyTorch doesn't need the module's CUDA toolkit at RUNTIME (only nvcc, which
# nothing here calls), so unset this and let PyTorch fall back to its own bundled
# libraries.
unset LD_LIBRARY_PATH

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN and no cached login at ~/.cache/huggingface/token -- run \`huggingface-cli login\` on this cluster (not just your laptop) or export HF_TOKEN before submitting}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

# 5. Run -- job-id-tagged output paths, same convention as jobs/train.sh
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/fanta_${SLURM_JOB_ID}.pt"
echo "Starting FANTA training with args: $@"
python3 train.py fanta \
    --use-wandb \
    --wandb-project fanta \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/fanta_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/fanta_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Training complete."
    # Held-out BOUQuET DEV was already scored periodically during training (see
    # fanta/train.py's epoch-boundary eval); this final job scores the
    # genuinely held-out TEST split exactly once, using the checkpoint training
    # just wrote -- no --dependency needed since training already finished by
    # the time this line runs. --output/--result-key included explicitly --
    # confirmed live (see jobs/train_superbpe.sh's own history) that omitting
    # them means the eval job only prints a report and writes NO mergeable
    # JSON, silently losing the ability to fold a real completed run into
    # results/all_tokenizers_comparison.json without a manual re-run.
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh fanta --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/fanta_comparison.json" --result-key fanta
else
    echo "Training failed." && exit 1
fi
