#!/bin/bash
#SBATCH --job-name=encoder_finetune
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# NER/POS/Taxi1500/SIB-200 finetune-then-zero-shot-transfer eval
# (systems/pretraining/encoder_cli_finetune.py, built on transformers.Trainer)
# for an encoder_train.py checkpoint. Single GPU -- Trainer handles its own
# device placement, no torchrun/DDP wiring needed here. SIB-200 (topic
# classification, mteb/sib200) is NOT part of Glot500's own eval suite --
# added on request, alongside the three Glot500 tasks that need a
# finetuned head.
#
# NO AUTO-RESUBMIT (unlike jobs/train_encoder_pretraining.sh): Trainer runs
# here with save_strategy="no" (see encoder_finetune_tagging.finetune_tagging/
# encoder_finetune_classification.finetune_classification's own docstrings)
# -- there is no intermediate checkpoint to resume a killed run from, so a
# run that exceeds --time just has to be resubmitted from scratch with a
# longer --time (or a smaller --max-train-examples) rather than resumed.
# NER/POS default to 10 epochs, Taxi1500/SIB-200 to 30 (Glot500's own
# evaluate_ner.sh/evaluate_pos.sh/zero_shot_train.py constants, reused for
# SIB-200 too since it's the same task shape as Taxi1500) -- widen --time or
# pass --max-train-examples/--max-eval-examples to cap dataset size for a
# faster smoke-test run before committing to a full one.
#
# --task ner/pos/sib200 pull WikiANN / Universal Dependencies / mteb/sib200
# live from the HF Hub (see encoder_cli_finetune.py's own module docstring
# for the exact verified repo ids/configs) -- needs HF_TOKEN. --task
# taxi1500 pulls English labels from GitHub directly (see
# encoder_finetune_taxi1500.py); non-English needs --taxi1500-eval-tsv
# pointing at a labeled file you've obtained yourself (Taxi1500-c is gated,
# not auto-downloadable) -- SIB-200 has no such gate, every one of its 206
# language configs is directly loadable.
#
# Usage:
#   sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --task ner --train-lang en --eval-lang de --output-dir finetune_out/ner_de
#   sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --task pos --train-config en_ewt --eval-config de_gsd --output-dir finetune_out/pos_de
#   sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --task taxi1500 --output-dir finetune_out/taxi1500
#   sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --task sib200 --train-lang-script eng_Latn --eval-lang-script deu_Latn --output-dir finetune_out/sib200_deu
#
# FULL EVAL across many languages in ONE run, no per-language retraining
# (--eval-langs/--eval-configs/--eval-lang-scripts for ner/pos/sib200
# respectively -- comma-separated, or the literal "all" for every config
# that Hub repo has, minus whichever language you trained on):
#   sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain/final.pt \
#       --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \
#       --task sib200 --train-lang-script eng_Latn --eval-lang-scripts all --output-dir finetune_out/sib200_full
#   # --max-eval-langs N caps an "all" sweep (prints exactly what got dropped, never silently truncates)
#
# All flags forward directly -- see `python3 -m systems.pretraining.encoder_cli_finetune --help`.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/rl-tokenizers
WORK_ROOT=/pfs/work9/workspace/scratch/tu_zxoqp65-rl-tokenizers  # larger-quota scratch -- see jobs/prep_pretraining_data.sh

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"
unset LD_LIBRARY_PATH  # avoids the cuda module's older cuDNN shadowing PyTorch's bundled one

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export HF_HOME=$WORK_ROOT/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs finetune_out data/taxi1500

echo "Starting encoder finetuning with args: $@"
python3 -m systems.pretraining.encoder_cli_finetune --device cuda "$@"

if [ $? -eq 0 ]; then
    echo "Finetuning complete."
else
    echo "Finetuning failed." && exit 1
fi
