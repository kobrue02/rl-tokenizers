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

# Regenerates results/all_tokenizers_comparison.json (scripts/combine_eval_results.py)
# and figures/tikz/ (scripts/generate_tikz_figures.py) from every results/*_comparison.json
# file that exists at run time. Meant to run as the final step of
# jobs/evaluate_latest_checkpoints.sh's own dependency chain (that script
# submits this with --dependency=afterany:<eval job ids> after fanning out
# one jobs/evaluate.sh job per tokenizer) -- but safe to submit directly any
# time you just want to regenerate figures from whatever comparison files
# already exist:
#   sbatch jobs/combine_and_generate_figures.sh
#
# Cheap: JSON merge + writing .tex/.dat text, no model loading, no network,
# no HF_HOME needed -- --time=00:15:00 is generous headroom, not a real
# benchmark need.
#
# Excludes results/all_tokenizers_comparison.json itself from its own input
# glob (see the loop below) -- otherwise a stale combined file would feed
# back into the NEW combine step as one more "source", at best redundant and
# at worst reintroducing a key a real per-tokenizer source has since dropped
# (e.g. a model removed from a later run).
#
# ALSO excludes any results/*indigenous_panel*_comparison.json -- confirmed
# live to be a REAL bug, not a hypothetical: results/indigenous_panel_comparison.json
# uses the SAME model-name keys as results/hf_frontier_comparison.json (both
# evaluate the same 33 repos) but a totally different results shape (see
# scripts/generate_tikz_figures.py's own load_indigenous_panel_rows docstring --
# {"combined": ..., "token_parity_by_anchor": ..., "morphology_spread": ...},
# no token_parity_spread at all). scripts/combine_eval_results.py's "later file in
# the list wins" merge silently overwrote every one of those 33 perfectly
# good hf_frontier entries with the wrong-shaped indigenous-panel version,
# which is what actually broke this job -- scripts/backfill_anchor_invariant_parity.py
# below had already confirmed hf_frontier_comparison.json itself was fine the
# whole time. The indigenous panel comparison has its own separate consumer
# (scripts/generate_tikz_figures.py --indigenous-panel) and was never meant to join
# the main BOUQuET-based comparison this job builds.
#
# Runs scripts/backfill_anchor_invariant_parity.py on every remaining input file
# before combining anyway (see that script's own docstring: computed purely
# from each file's own already-stored token_parity, no re-tokenization or
# network calls, and a no-op for any entry that already has both fields) --
# scripts/generate_tikz_figures.py's load_rows hard-requires token_parity_spread on
# every entry and raises ValueError otherwise. Kept as a real safety net for
# whatever future results file genuinely does predate this metric, even
# though it turned out not to be the cause of this particular failure.

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
