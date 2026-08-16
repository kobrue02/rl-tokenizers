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

# Regenerates results/all_tokenizers_comparison.json (combine_eval_results.py)
# and figures/tikz/ (generate_tikz_figures.py) from every results/*_comparison.json
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
    inputs+=("$f")
done

if [ "${#inputs[@]}" -eq 0 ]; then
    echo "No results/*_comparison.json files found -- nothing to combine." >&2
    exit 1
fi

echo "Combining ${#inputs[@]} result file(s): ${inputs[*]}"
python3 combine_eval_results.py --input "${inputs[@]}" --output results/all_tokenizers_comparison.json
if [ $? -ne 0 ]; then
    echo "combine_eval_results.py failed." && exit 1
fi

echo "Generating figures from results/all_tokenizers_comparison.json ..."
python3 generate_tikz_figures.py --input results/all_tokenizers_comparison.json --output-dir figures/tikz
if [ $? -eq 0 ]; then
    echo "Figures regenerated under figures/tikz/."
else
    echo "generate_tikz_figures.py failed." && exit 1
fi
