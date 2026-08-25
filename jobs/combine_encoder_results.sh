#!/bin/bash
#SBATCH --job-name=combine_encoder_results
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Regenerates results/encoder_comparison.json and results/encoder_comparison.md
# from every results/encoder/*.json that exists at run time -- mirrors
# jobs/combine_and_generate_figures.sh's own role for the tokenizer-eval
# side. Usually the final step of jobs/run_encoder_eval_suite.sh's
# dependency chain, but safe to run directly any time:
#   sbatch jobs/combine_encoder_results.sh
# Cheap (JSON merge + markdown generation, no model/network) -- 15min is
# generous headroom.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers

module load devel/python/3.13.3-llvm-19.1
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs results/encoder

shopt -s nullglob
INPUTS=(results/encoder/*.json)
shopt -u nullglob

if [ "${#INPUTS[@]}" -eq 0 ]; then
    echo "No results/encoder/*.json files found -- nothing to combine." >&2
    exit 1
fi

echo "Combining ${#INPUTS[@]} result file(s): ${INPUTS[*]}"
python3 -m scripts.combine_encoder_results --input "${INPUTS[@]}" --output results/encoder_comparison.json
python3 -m scripts.generate_encoder_comparison_table --input results/encoder_comparison.json --output results/encoder_comparison.md

echo "Wrote results/encoder_comparison.json and results/encoder_comparison.md"
cat results/encoder_comparison.md
