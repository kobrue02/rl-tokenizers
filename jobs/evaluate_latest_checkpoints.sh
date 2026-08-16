#!/bin/bash
# Finds the most recently modified checkpoint for each of this repo's OWN
# trained tokenizers (fairtok/magnet/flexitokens/manta/fanta/superbpe/bpe --
# NOT hf_frontier/claude_tokenizer, which aren't trained checkpoints this
# project fits, see evaluate.py's own TOKENIZERS dict comment) and submits
# one jobs/evaluate.sh job per tokenizer against it.
#
# Deliberately NOT a single SLURM job running all seven evaluations
# sequentially: jobs/evaluate.sh's own --time=08:00:00 is already a
# generous-but-unbenchmarked estimate for ONE tokenizer's full BOUQuET-test
# pass (259 langs, ~272k rows, unbatched per-document scoring for the five
# neural systems -- see that script's own comment, confirmed to already need
# close to that long for a single FANTA run). Chaining seven of those in one
# job could mean up to ~56 hours in one allocation -- almost certainly over
# any real per-job time limit, with no way to save/resume partial progress
# across tokenizers if it got killed partway. Submitting seven separate jobs
# instead lets SLURM run/queue them independently and in parallel -- exactly
# what jobs/evaluate.sh was already designed for (every train_*.sh script
# already ends by submitting exactly one of these for its own tokenizer).
#
# This script itself does no computation -- just checkpoint discovery +
# sbatch submission -- so it's meant to be run directly on the login node
# (bash jobs/evaluate_latest_checkpoints.sh), not itself submitted via sbatch.
#
# WIRED INTO FIGURE GENERATION: each evaluate.sh call below is given
# --output results/<tokenizer>_comparison.json (--result-key defaults to the
# tokenizer's own system_label -- see common.eval.cross_tokenizer.
# build_eval_arg_parser's own docstring -- exactly the {result_key: results}
# shape scripts/combine_eval_results.py already expects, the SAME pipeline
# hf_frontier/claude_tokenizer's own evaluate.py runs already write into).
# Previously these seven systems' evaluate.sh calls never passed --output at
# all, so their results only ever printed to logs -- they never actually
# reached scripts/combine_eval_results.py/scripts/generate_tikz_figures.py. Once every
# submitted job finishes (successfully or not -- afterany, not afterok: a
# partial batch's real results shouldn't be blocked from reaching the
# figures just because one tokenizer's eval job failed), a final
# jobs/combine_and_generate_figures.sh job runs automatically to regenerate
# results/all_tokenizers_comparison.json and figures/tikz/ from every
# *_comparison.json file that actually exists at that point -- not just the
# ones this run touched, see that script's own comment.
#
# Usage:
#   bash jobs/evaluate_latest_checkpoints.sh
#   bash jobs/evaluate_latest_checkpoints.sh --eval-data-source bouquet   # dev, not test -- cheaper pass
#   bash jobs/evaluate_latest_checkpoints.sh --num-groups 50              # cheaper exploratory pass
#   # any extra args are forwarded to every jobs/evaluate.sh call, after
#   # --checkpoint <latest> --eval-data-source bouquet_test --output
#   # results/<tok>_comparison.json -- argparse takes the LAST value for a
#   # repeated flag, so e.g. passing your own --eval-data-source above
#   # overrides the bouquet_test default.

set -uo pipefail

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
cd "$PROJECT_ROOT"

# tokenizer name -> checkpoint filename glob, matching each train_*.sh
# script's own CHECKPOINT_PATH convention exactly (fairtok's is "policy_",
# not "fairtok_" -- see jobs/train.sh; bpe is the only one that's .json,
# every neural system is .pt -- see jobs/train_*.sh's own CHECKPOINT_PATH
# lines).
declare -A PATTERNS=(
    [fairtok]="policy_*.pt"
    [magnet]="magnet_*.pt"
    [flexitokens]="flexitokens_*.pt"
    [manta]="manta_*.pt"
    [fanta]="fanta_*.pt"
    [superbpe]="superbpe_*.pt"
    [bpe]="bpe_*.json"
)

echo "Discovering latest checkpoints under checkpoints/ ..."
JOB_IDS=()
for tok in "${!PATTERNS[@]}"; do
    pattern="${PATTERNS[$tok]}"
    # ls -t sorts by mtime, newest first. No checkpoint yet for this
    # tokenizer is expected (not every system has necessarily finished
    # training), so this is handled as a skip, not a hard failure --
    # deliberately not using `set -e` for that reason.
    latest=$(ls -t checkpoints/$pattern 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "  $tok: no checkpoint found matching checkpoints/$pattern -- skipping"
        continue
    fi
    echo "  $tok: $latest"
    # --parsable makes sbatch print ONLY the numeric job id (no "Submitted
    # batch job" prose), so it can be collected directly into JOB_IDS below
    # for the dependency chain.
    jobid=$(sbatch --parsable jobs/evaluate.sh "$tok" --checkpoint "$latest" \
        --eval-data-source bouquet_test --output "results/${tok}_comparison.json" "$@")
    echo "    submitted job $jobid"
    JOB_IDS+=("$jobid")
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "No checkpoints found for any tokenizer -- nothing submitted, figures not regenerated."
    exit 0
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
FIGJOB=$(sbatch --parsable --dependency="afterany:$DEP" jobs/combine_and_generate_figures.sh)
echo "Submitted ${#JOB_IDS[@]} evaluation job(s): ${JOB_IDS[*]}"
echo "Figures will regenerate automatically once they finish -- job $FIGJOB (dependency=afterany:$DEP)"
