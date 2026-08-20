#!/bin/bash
# Finds the most recent checkpoint for each of this repo's own trained
# tokenizers (fairtok/magnet/flexitokens/manta/fanta/superbpe/bpe) and
# submits one jobs/evaluate.sh job per tokenizer -- separate jobs, not one
# chained job, since 7x evaluate.sh's own ~8h budget would exceed any real
# time limit with no way to resume partway. Run directly on the login node
# (not itself submitted via sbatch).
#
# Each job gets --output results/<tokenizer>_comparison.json, the shape
# scripts/combine_eval_results.py expects; once every submitted job finishes
# (afterany, not afterok -- one tokenizer's failure shouldn't block the
# rest), jobs/combine_and_generate_figures.sh runs automatically to
# regenerate results/all_tokenizers_comparison.json and figures/tikz/.
#
# Usage:
#   bash jobs/evaluate_latest_checkpoints.sh
#   bash jobs/evaluate_latest_checkpoints.sh --num-groups 50   # cheaper exploratory pass
#   # extra args forward to every jobs/evaluate.sh call (argparse takes the
#   # last value for a repeated flag, so you can override --eval-data-source).

set -uo pipefail

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
cd "$PROJECT_ROOT"

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
)

echo "Discovering latest checkpoints under checkpoints/ ..."
JOB_IDS=()
for tok in "${!PATTERNS[@]}"; do
    pattern="${PATTERNS[$tok]}"
    latest=$(ls -t checkpoints/$pattern 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "  $tok: no checkpoint found matching checkpoints/$pattern -- skipping"
        continue
    fi
    echo "  $tok: $latest"
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
