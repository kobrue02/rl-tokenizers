#!/bin/bash
#SBATCH --job-name=contamination_check
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# n-gram overlap check between a pretraining corpus and one eval benchmark's
# own examples (systems/pretraining/cli_contamination.py). CPU-bound, text
# only -- no tokenization/GPU, so cpu_il + a modest core count is enough.
#
# PREREQUISITE for --corpus-dataset glot500: reads from the same one-time
# local disk cache jobs/prep_pretraining_data.sh needs (jobs/prepare_glot500.sh
# must have completed first) -- NOT live HF streaming. The benchmark side
# (xnli/xcopa/flores_mt/blimp/cola/squad) still loads live from the HF Hub
# regardless, so HF_TOKEN/network access is still needed for that half.
#
# NO RESUME: unlike jobs/train_pretraining.sh/prep_pretraining_data.sh, the
# underlying scan (contamination.py) has no checkpoint of its own -- a run
# killed by the time limit has nothing to resume from and a resubmit
# restarts the corpus scan from document 0. Size --max-corpus-docs and
# --time together so ONE run actually finishes; --time=12:00:00 above is a
# starting guess, not a measured number -- tighten or widen it once you've
# seen a real run's own throughput. This is a deliberate scope choice, not
# an oversight: contamination.py's own module docstring already frames this
# as an on-demand check a user runs and inspects, not a queued pipeline
# stage with resumability semantics like the training/prep jobs have.
#
# --corpus-langs uses glot500's own LOCAL bare-code filenames (e.g. "eng",
# confirmed against a real data/glot500/eng.jsonl), NOT a benchmark's own
# language-code convention -- XNLI's "sw" (ISO 639-1) is NOT the same
# string as glot500's own Swahili file. Once jobs/prepare_glot500.sh has
# finished, find the REAL local codes with:
#   python3 -c "from common.data.corpora import list_glot500_local_langs; print(sorted(list_glot500_local_langs()))"
# rather than assuming a benchmark's own --benchmark-langs codes carry over.
#
# Usage (English-only benchmarks -- cola/squad/blimp -- scoped to glot500's
# own English config, since that's the realistic contamination risk for them):
#   sbatch jobs/check_contamination.sh \
#       --benchmark cola --corpus-dataset glot500 --corpus-langs eng \
#       --output results/contamination_cola_glot500.json
#
# Usage (a multilingual benchmark -- xnli/xcopa/flores_mt -- "all" scans
# every locally-prepared glot500 language; substitute a real subset from
# list_glot500_local_langs above if you want to scope this to just the
# --benchmark-langs you're checking):
#   sbatch jobs/check_contamination.sh \
#       --benchmark xnli --benchmark-langs en,de,fr,sw,zh \
#       --corpus-dataset glot500 --corpus-langs all \
#       --output results/contamination_xnli_glot500.json
#
# All flags forward directly to systems/pretraining/cli_contamination.py --
# see `python3 -m systems.pretraining.cli_contamination --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

# 44TiB Lustre scratch -- glot500's local cache lives here (see
# jobs/prepare_glot500.sh/jobs/prep_pretraining_data.sh's own comments on
# this same split), not $HOME's 550GiB hard cap.
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers

module load devel/python/3.13.3-llvm-19.1

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

echo "Starting contamination check with args: $@"
python3 -m systems.pretraining.cli_contamination "$@"
