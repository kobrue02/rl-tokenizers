#!/bin/bash
#SBATCH --job-name=bpe_train
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Standard byte-level BPE baseline -- wraps HuggingFace's `tokenizers` library
# directly (see bpe/model.py's module docstring for why this one, unlike every
# other baseline in this repo, doesn't reimplement its own algorithm). No GPU
# is requested: fitting is a single-shot, Rust-backed corpus-statistics fit,
# not gradient descent -- cpu_il is the correct partition, same reasoning as
# jobs/train_superbpe.sh.
#
# --time=01:00:00: benchmarked directly (not just estimated) -- fitting to
# vocab_size=5000 over an 8000-sentence/~320KB corpus took 0.25s, roughly two
# orders of magnitude faster than superbpe/'s pure-Python trainer at a
# comparable scale (Rust vs. Python doing the same kind of work). A full
# --vocab-size 50000 run over the real pooled corpus should still finish in
# minutes, not hours; 1 hour is a generous margin, not a tight estimate.
#
# Usage:
#   sbatch jobs/train_bpe.sh --data-source all --langs all --vocab-size 50000
#   sbatch jobs/train_bpe.sh --data-source oldi_seed --vocab-size 2000   # quicker, single source
#
# All train.py bpe / bpe.cli flags are forwarded directly -- see
# `python train.py bpe --help`.
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
module load devel/python/3.13.3-llvm-19.1

# 3. Environment
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
: "${HF_TOKEN:?No HF_TOKEN and no cached login at ~/.cache/huggingface/token -- run \`huggingface-cli login\` on this cluster (not just your laptop) or export HF_TOKEN before submitting}"
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# 4. Project
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs checkpoints vocab_out

# 5. Run -- job-id-tagged output paths, same convention as jobs/train.sh.
# NOTE: the checkpoint extension is .json, not .pt -- bpe/inference.py loads
# it via tokenizers.Tokenizer.from_file(), the library's own native
# serialization, not torch.load (see that module's docstring for why).
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/bpe_${SLURM_JOB_ID}.json"
echo "Starting BPE fitting with args: $@"
python3 train.py bpe \
    --use-wandb \
    --wandb-project bpe \
    --run-name "slurm-${SLURM_JOB_ID}" \
    --output-dir "$CHECKPOINT_PATH" \
    --vocab-out "$PROJECT_ROOT/vocab_out/bpe_vocab_${SLURM_JOB_ID}.json" \
    --vocab-stats-out "$PROJECT_ROOT/vocab_out/bpe_vocab_stats_${SLURM_JOB_ID}.json" \
    "$@"

if [ $? -eq 0 ]; then
    echo "Fitting complete."
    # --output/--result-key included explicitly -- confirmed live (see
    # jobs/train_superbpe.sh's own history) that omitting them means the
    # eval job only prints a report and writes NO mergeable JSON, silently
    # losing the ability to fold a real completed run into
    # results/all_tokenizers_comparison.json without a manual re-run.
    echo "Submitting final test-set evaluation job..."
    sbatch jobs/evaluate.sh bpe --checkpoint "$CHECKPOINT_PATH" --eval-data-source bouquet_test \
        --output "results/bpe_comparison.json" --result-key bpe
else
    echo "Fitting failed." && exit 1
fi
