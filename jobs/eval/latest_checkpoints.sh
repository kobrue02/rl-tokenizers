#!/bin/bash
# THE single entrypoint for "evaluate every tokenizer this project compares":
# this project's own trained tokenizers (fairtok/magnet/flexitokens/manta/
# fanta/superbpe/bpe/parity_bpe), the external hf_frontier panel
# (configs/eval/hf_frontier.yml), and BLT's entropy-based patcher
# (configs/eval/blt.yml). Submits one jobs/evaluate*.sh job per system --
# separate jobs, not one chained job, since their combined time budgets
# (BLT alone is ~16-20h) would exceed any real time limit with no way to
# resume partway. Run directly on the login node (not itself submitted via
# sbatch).
#
# SKIP-IF-UP-TO-DATE: before submitting each job, checks whether its output
# file already exists AND is newer than whatever would make it stale (a
# checkpoint file for the project's own tokenizers, the relevant configs/*.yml
# for hf_frontier/blt) -- mtime-based, same convention as `make`. This matters
# most for BLT (a ~16-20h job you do NOT want re-submitted by accident every
# time this script runs) but applies uniformly. To force a re-run of
# something already up to date, either `rm` its results/*_comparison.json
# first or `touch` the newer source (checkpoint/config) it should react to.
#
# Each job gets --output results/<tokenizer>_comparison.json, the shape
# scripts/combine_eval_results.py expects; once every submitted job finishes
# (afterany, not afterok -- one tokenizer's failure shouldn't block the
# rest), jobs/combine/combine_and_generate_figures.sh runs automatically to
# regenerate results/all_tokenizers_comparison.json and figures/tikz/. If
# EVERY system is already up to date, nothing is submitted and figures are
# NOT regenerated (nothing changed that would change them) -- run
# `sbatch jobs/combine/combine_and_generate_figures.sh` directly if you want to force
# a rebuild anyway (e.g. after editing scripts/generate_tikz_figures.py itself).
#
# Usage:
#   bash jobs/eval/latest_checkpoints.sh
#   bash jobs/eval/latest_checkpoints.sh --num-groups 50   # cheaper exploratory pass
#   # extra args forward to every jobs/eval/evaluate.sh call for this project's OWN
#   # tokenizers only (argparse takes the last value for a repeated flag, so
#   # you can override --eval-data-source) -- hf_frontier/blt read their own
#   # flags from configs/eval/hf_frontier.yml / configs/eval/blt.yml instead,
#   # since their arg shapes differ (--hf-repo-id, no --checkpoint, etc.) and
#   # forwarding the same "$@" to all three would silently break two of them.

set -uo pipefail

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
cd "$PROJECT_ROOT"

JOB_IDS=()

# submit_if_stale name output_path reference_path -- sbatch_args...
# Returns (via JOB_IDS) whether a job was actually submitted. `reference_path`
# missing entirely (e.g. a checkpoint glob that matched nothing) is handled
# by the caller before this is invoked -- this function only compares mtimes
# once both paths are known to exist.
submit_if_stale() {
    local name="$1" output="$2" reference="$3"
    shift 3
    if [ -f "$output" ] && [ "$output" -nt "$reference" ]; then
        echo "  $name: $output is up to date (newer than $reference) -- skipping"
        return
    fi
    local jobid
    jobid=$(sbatch --parsable "$@")
    echo "  $name: submitted job $jobid (output=$output will be older than $reference until it finishes)"
    JOB_IDS+=("$jobid")
}

echo "=== This project's own trained tokenizers ==="
# tokenizer -> checkpoint glob, matching each train_*.sh's own CHECKPOINT_PATH
# (fairtok's is "policy_", not "fairtok_"; bpe is .json, everything else is .pt).
declare -A PATTERNS=(
    [fairtok]="policy_*.pt"
    [magnet]="magnet_*.pt"
    [flexitokens]="flexitokens_*.pt"
    [manta]="manta_*.pt"
    [fanta]="fanta_*.pt"
    [superbpe]="superbpe_*.pt"
    [bpe]="bpe_*.json"
    [parity_bpe]="parity_bpe_*.json"
)
for tok in "${!PATTERNS[@]}"; do
    pattern="${PATTERNS[$tok]}"
    latest=$(ls -t checkpoints/$pattern 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "  $tok: no checkpoint found matching checkpoints/$pattern -- skipping"
        continue
    fi
    submit_if_stale "$tok" "results/${tok}_comparison.json" "$latest" \
        jobs/eval/evaluate.sh "$tok" --checkpoint "$latest" \
        --eval-data-source bouquet_test --output "results/${tok}_comparison.json" "$@"
done

echo "=== External hf_frontier panel ==="
submit_if_stale "hf_frontier" "results/hf_frontier_comparison.json" "configs/eval/hf_frontier.yml" \
    jobs/eval/hf_frontier.sh -c configs/eval/hf_frontier.yml

echo "=== BLT (entropy-based dynamic patching) ==="
# ~16-20h job (see jobs/eval/blt.sh's own comment) -- the whole reason
# this script needs skip-if-up-to-date at all, not just for BLT's own sake.
submit_if_stale "blt" "results/blt_comparison.json" "configs/eval/blt.yml" \
    jobs/eval/blt.sh -c configs/eval/blt.yml

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "Nothing to submit -- every system's results are already up to date."
    echo "(Run 'sbatch jobs/combine/combine_and_generate_figures.sh' directly if you want to force-regenerate figures anyway.)"
    exit 0
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
FIGJOB=$(sbatch --parsable --dependency="afterany:$DEP" jobs/combine/combine_and_generate_figures.sh)
echo "Submitted ${#JOB_IDS[@]} evaluation job(s): ${JOB_IDS[*]}"
echo "Figures will regenerate automatically once they finish -- job $FIGJOB (dependency=afterany:$DEP)"
