#!/bin/bash
#SBATCH --job-name=combine_and_figures
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Regenerates results/all_tokenizers_comparison.json and figures/tikz/ from
# every results/*_comparison.json that exists at run time. Usually the final
# step of jobs/evaluate_latest_checkpoints.sh's dependency chain, but safe to
# run directly any time: sbatch jobs/combine_and_generate_figures.sh
# Cheap (JSON merge + text generation, no model/network) -- 15min is generous headroom.
#
# Excludes results/all_tokenizers_comparison.json itself (avoids feeding a
# stale combined file back into its own next combine) and any
# *indigenous_panel*_comparison.json -- CONFIRMED LIVE BUG: that file reuses
# the SAME model-name keys as hf_frontier_comparison.json but a totally
# different shape, and combine_eval_results.py's "later file wins" merge
# silently overwrote 33 good hf_frontier entries with the wrong shape. The
# indigenous panel has its own separate consumer
# (scripts/generate_tikz_figures.py --indigenous-panel) and must stay excluded here.
#
# Runs scripts/backfill_anchor_invariant_parity.py first (a no-op for
# entries that already have token_parity_spread) since generate_tikz_figures.py
# hard-requires that field on every entry.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

module load devel/python/3.13.3-llvm-19.1
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results figures/tikz

shopt -s nullglob
inputs=()
for f in results/*_comparison.json; do
    [ "$f" = "results/all_tokenizers_comparison.json" ] && continue
    [[ "$f" == *indigenous_panel* ]] && continue
    inputs+=("$f")
done

if [ "${#inputs[@]}" -eq 0 ]; then
    echo "No results/*_comparison.json files found -- nothing to combine." >&2
    exit 1
fi

echo "Backfilling token_parity_gm/token_parity_spread where missing ..."
python3 -m scripts.backfill_anchor_invariant_parity --input "${inputs[@]}"
if [ $? -ne 0 ]; then
    echo "scripts/backfill_anchor_invariant_parity.py failed." && exit 1
fi

echo "Combining ${#inputs[@]} result file(s): ${inputs[*]}"
python3 -m scripts.combine_eval_results --input "${inputs[@]}" --output results/all_tokenizers_comparison.json
if [ $? -ne 0 ]; then
    echo "scripts/combine_eval_results.py failed." && exit 1
fi

echo "Generating figures from results/all_tokenizers_comparison.json ..."
python3 -m scripts.generate_tikz_figures --input results/all_tokenizers_comparison.json --output-dir figures/tikz
if [ $? -eq 0 ]; then
    echo "Figures regenerated under figures/tikz/."
else
    echo "scripts/generate_tikz_figures.py failed." && exit 1
fi
