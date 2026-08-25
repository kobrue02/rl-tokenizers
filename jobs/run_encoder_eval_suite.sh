#!/bin/bash
# Submits the FULL encoder eval + finetune suite -- pseudoperplexity,
# sentence retrieval, roundtrip alignment, and NER/POS/Taxi1500/SIB-200
# finetuning (each with a --eval-langs=all-style sweep across every target
# language) -- for every tokenizer listed in the arrays below, then chains
# jobs/combine_encoder_results.sh to run automatically once everything
# finishes. Separate jobs, not one chained job (mirrors
# jobs/evaluate_latest_checkpoints.sh's own reasoning): one tokenizer/
# benchmark's own budget could exceed any single time limit with no way to
# resume partway, and one failure shouldn't block the rest. Run directly on
# the login node (not itself submitted via sbatch).
#
# 7 jobs PER tokenizer (2 tokenizers below = 14 total): pppl, retrieval,
# roundtrip, ner, pos, taxi1500, sib200.
#
# PREREQUISITE for roundtrip: bible_nlp prepared locally first (sbatch
# jobs/prepare_bible_nlp.sh) -- this script checks for data/bible_nlp and
# skips roundtrip (with a warning, for every tokenizer) rather than
# submitting a job that would just fail immediately.
#
# MAX_EVAL_LANGS below caps the ner/pos/sib200 --eval-langs/--eval-configs/
# --eval-lang-scripts=all sweep (WikiANN has 176 configs, Universal
# Dependencies 350 -- a genuinely unbounded first run risks exceeding
# finetune_encoder.sh's own --time before finishing, and NO AUTO-RESUBMIT
# exists for finetuning -- see ENCODER.md). Prints exactly what's being
# capped to, never silently. Set to 0 to disable the cap once you've
# confirmed a full run's real wall-clock budget from a capped one.
#
# Edit the ENCODER_CHECKPOINTS/TOKENIZER_SYSTEMS/TOKENIZER_CHECKPOINTS/
# TOKENIZER_VOCAB_JSONS arrays below to add more tokenizers/checkpoints --
# the same label must be a key in ALL FOUR, and becomes both --label (see
# scripts.combine_encoder_results) and the results/encoder/*.json filename.
#
# Usage:
#   bash jobs/run_encoder_eval_suite.sh
# All --benchmark/--task language settings (--pair, --lang, --train-lang,
# etc.) are the script-level variables below, not CLI args to this script
# -- edit them directly for a different eval language/pair/config.

set -uo pipefail

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
cd "$PROJECT_ROOT"

# label -> encoder_train.py checkpoint (--checkpoint)
declare -A ENCODER_CHECKPOINTS=(
    [bpe]="checkpoints/encoder_pretrain_bpe/final.pt"
    [fanta]="checkpoints/encoder_pretrain_fanta/final.pt"
)
# label -> --system
declare -A TOKENIZER_SYSTEMS=(
    [bpe]="bpe"
    [fanta]="fanta"
)
# label -> --tokenizer-checkpoint
declare -A TOKENIZER_CHECKPOINTS=(
    [bpe]="checkpoints/bpe_50k.json"
    [fanta]="checkpoints/fanta_6284655.pt"
)
# label -> --vocab-json ("" for native systems that don't need one, e.g. bpe/superbpe)
declare -A TOKENIZER_VOCAB_JSONS=(
    [bpe]=""
    [fanta]="vocab_out/fanta_vocab_6284655.json"
)

# --- eval-task language settings -- see `--help` on each CLI for what these mean ---
RETRIEVAL_DATASET=tatoeba_mt
RETRIEVAL_PAIR=deu-eng
RETRIEVAL_SPLIT=test
PPPL_LANG=deu
TRAIN_LANG=en               # ner
TRAIN_CONFIG=en_ewt         # pos
TRAIN_LANG_SCRIPT=eng_Latn  # sib200
MAX_EVAL_LANGS=20           # see MAX_EVAL_LANGS comment above; 0 disables the cap
FINETUNE_SWEEP_TIME=12:00:00  # ner/pos/sib200's own --eval-*=all sweep needs more
# than finetune_encoder.sh's own 6h default even WITH the cap above (many extra
# HF dataset downloads); taxi1500 (English-only, no sweep) keeps the 6h default.

mkdir -p logs results/encoder
JOB_IDS=()

ROUNDTRIP_OK=1
if [ ! -d "data/bible_nlp" ]; then
    ROUNDTRIP_OK=0
    echo "data/bible_nlp not found locally -- skipping roundtrip alignment for every" \
         "tokenizer (run 'sbatch jobs/prepare_bible_nlp.sh' first if you want it included)."
fi

MAX_EVAL_LANGS_ARGS=()
if [ "$MAX_EVAL_LANGS" -gt 0 ]; then
    MAX_EVAL_LANGS_ARGS=(--max-eval-langs "$MAX_EVAL_LANGS")
    echo "Capping ner/pos/sib200 --eval-*=all sweeps to $MAX_EVAL_LANGS languages" \
         "each (set MAX_EVAL_LANGS=0 in this script to run the true, unbounded sweep)."
fi

for label in "${!ENCODER_CHECKPOINTS[@]}"; do
    checkpoint="${ENCODER_CHECKPOINTS[$label]}"
    if [ ! -f "$checkpoint" ]; then
        echo "$label: no checkpoint at $checkpoint -- skipping entirely"
        continue
    fi
    echo "$label ($checkpoint):"

    COMMON_ARGS=(--checkpoint "$checkpoint" --system "${TOKENIZER_SYSTEMS[$label]}" \
        --tokenizer-checkpoint "${TOKENIZER_CHECKPOINTS[$label]}" --label "$label")
    if [ -n "${TOKENIZER_VOCAB_JSONS[$label]}" ]; then
        COMMON_ARGS+=(--vocab-json "${TOKENIZER_VOCAB_JSONS[$label]}")
    fi

    jobid=$(sbatch --parsable jobs/evaluate_encoder.sh "${COMMON_ARGS[@]}" \
        --benchmark pppl --dataset "$RETRIEVAL_DATASET" --pair "$RETRIEVAL_PAIR" \
        --split "$RETRIEVAL_SPLIT" --lang "$PPPL_LANG" --output "results/encoder/pppl_${label}.json")
    echo "  pppl: job $jobid"; JOB_IDS+=("$jobid")

    jobid=$(sbatch --parsable jobs/evaluate_encoder.sh "${COMMON_ARGS[@]}" \
        --benchmark retrieval --dataset "$RETRIEVAL_DATASET" --pair "$RETRIEVAL_PAIR" \
        --split "$RETRIEVAL_SPLIT" --output "results/encoder/retrieval_${label}.json")
    echo "  retrieval: job $jobid"; JOB_IDS+=("$jobid")

    if [ "$ROUNDTRIP_OK" -eq 1 ]; then
        jobid=$(sbatch --parsable jobs/evaluate_encoder.sh "${COMMON_ARGS[@]}" \
            --benchmark roundtrip --output "results/encoder/roundtrip_${label}.json")
        echo "  roundtrip: job $jobid"; JOB_IDS+=("$jobid")
    fi

    jobid=$(sbatch --parsable --time="$FINETUNE_SWEEP_TIME" jobs/finetune_encoder.sh "${COMMON_ARGS[@]}" \
        --task ner --train-lang "$TRAIN_LANG" --eval-langs all "${MAX_EVAL_LANGS_ARGS[@]}" \
        --output-dir "finetune_out/ner_${label}" --results-output "results/encoder/ner_${label}.json")
    echo "  ner: job $jobid"; JOB_IDS+=("$jobid")

    jobid=$(sbatch --parsable --time="$FINETUNE_SWEEP_TIME" jobs/finetune_encoder.sh "${COMMON_ARGS[@]}" \
        --task pos --train-config "$TRAIN_CONFIG" --eval-configs all "${MAX_EVAL_LANGS_ARGS[@]}" \
        --output-dir "finetune_out/pos_${label}" --results-output "results/encoder/pos_${label}.json")
    echo "  pos: job $jobid"; JOB_IDS+=("$jobid")

    jobid=$(sbatch --parsable jobs/finetune_encoder.sh "${COMMON_ARGS[@]}" \
        --task taxi1500 \
        --output-dir "finetune_out/taxi1500_${label}" --results-output "results/encoder/taxi1500_${label}.json")
    echo "  taxi1500: job $jobid"; JOB_IDS+=("$jobid")

    jobid=$(sbatch --parsable --time="$FINETUNE_SWEEP_TIME" jobs/finetune_encoder.sh "${COMMON_ARGS[@]}" \
        --task sib200 --train-lang-script "$TRAIN_LANG_SCRIPT" --eval-lang-scripts all "${MAX_EVAL_LANGS_ARGS[@]}" \
        --output-dir "finetune_out/sib200_${label}" --results-output "results/encoder/sib200_${label}.json")
    echo "  sib200: job $jobid"; JOB_IDS+=("$jobid")
done

if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "No encoder checkpoints found -- nothing submitted, results not combined."
    exit 0
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
COMBINEJOB=$(sbatch --parsable --dependency="afterany:$DEP" jobs/combine_encoder_results.sh)
echo "Submitted ${#JOB_IDS[@]} job(s): ${JOB_IDS[*]}"
echo "results/encoder_comparison.{json,md} will regenerate automatically once they finish -- job $COMBINEJOB (dependency=afterany:$DEP)"
