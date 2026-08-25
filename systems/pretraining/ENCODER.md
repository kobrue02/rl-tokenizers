# Encoder pipeline (MLM, XLM-R architecture)

A from-scratch, randomly-initialized XLM-R-architecture bidirectional
encoder (`encoder_model.py`) trained with MLM on the exact same packed
token shards and tokenizer systems the decoder pipeline (`train.py`,
`data_prep.py`) already uses — see `encoder_model_configs.py`'s own
docstring for why this is from-scratch rather than continued-pretraining
real XLM-R/Glot500-m weights (fair "same architecture, different
tokenizer" comparison against the rest of this project's own experiments).

Every CLI below accepts `-c`/`--config path/to/file.yml` — see
`configs/README.md` for the full precedence rules (same as every other
CLI in this repo). Run `python3 -m systems.pretraining.<module> --help`
for the complete flag list of any command; this file only covers the
shape of the pipeline and the gotchas worth knowing before you run it.

## 1. Pretraining

```bash
sbatch jobs/train_encoder_pretraining.sh \
    --shard-dir pretrain_data/glot500_bpe \
    --output-dir checkpoints/encoder_pretrain_bpe \
    --encoder-size base --total-steps 50000 --seq-len 512 --per-device-batch-size 16
```

- `--shard-dir` points at the SAME shards `data_prep.py` builds for the
  decoder — nothing about shard construction differs between training a
  decoder or this encoder over the same corpus/tokenizer. The tokenizer's
  identity is already recorded in that directory's `shards_meta.json`; you
  don't pass `--system`/`--checkpoint` for training itself.
- **Give every run its own `--output-dir`.** It defaults to the same
  `checkpoints/encoder_pretrain` regardless of `--shard-dir` — two runs
  left at the default (e.g. bpe and fanta) will overwrite each other's
  checkpoints.
- DDP, not FSDP (`encoder_train.py`'s own module docstring explains why —
  the model is small enough that full replication is fine up to at least
  4x A100). Multi-GPU: pass `--gres=gpu:N` and scale `--cpus-per-task`
  with it (`--num-workers` DataLoader workers spawn *per rank*).
  `--per-device-batch-size` is PER GPU — effective batch size is
  `per_device_batch_size × grad_accum_steps × world_size`, so more GPUs
  alone already multiplies your effective batch.

## 2. Eval: pseudoperplexity / retrieval / roundtrip alignment

The three of Glot500's six eval tasks that don't need a finetuned head
(see `encoder_eval.py`'s own docstring for why NER/POS/Taxi1500 are
separate). These need the actual tokenizer (to encode real text), unlike
training:

```bash
sbatch jobs/evaluate_encoder.sh --checkpoint checkpoints/encoder_pretrain_bpe/final.pt \
    --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
    --benchmark retrieval --dataset tatoeba_mt --pair deu-eng --split test

sbatch jobs/evaluate_encoder.sh --checkpoint ... --system bpe --tokenizer-checkpoint ... \
    --benchmark roundtrip --cycle-langs eng,fra,deu,eng

sbatch jobs/evaluate_encoder.sh --checkpoint ... --system bpe --tokenizer-checkpoint ... \
    --benchmark pppl --dataset tatoeba_mt --pair deu-eng --split test --lang deu
```

- Span-family tokenizers (`fairtok`/`magnet`/`flexitokens`/`manta`/`fanta`)
  need `--vocab-json` too, same requirement as everywhere else in this repo.
- `--benchmark roundtrip` (and any `--dataset bible_nlp` run) needs
  `bible_nlp` prepared locally FIRST — `sbatch jobs/prepare_bible_nlp.sh` —
  it has no live-streaming fallback.

## 3. Finetuning: NER / POS / Taxi1500 / SIB-200

The tasks that DO need a finetuned head (`transformers.Trainer` under the
hood — see `encoder_finetune.py`'s own docstring). Each dataset uses a
different language-code convention, unfortunately not a choice made here:

| `--task` | tokenizer flag | code format | example |
|---|---|---|---|
| `ner` | `--train-lang`/`--eval-lang` | 2-letter | `de` |
| `pos` | `--train-config`/`--eval-config` | `{lang}_{treebank}` | `de_gsd` |
| `sib200` | `--train-lang-script`/`--eval-lang-script` | `{lang3}_{Script}` | `deu_Latn` |
| `taxi1500` | (English-only auto-download; see below) | — | — |

```bash
sbatch jobs/finetune_encoder.sh --checkpoint checkpoints/encoder_pretrain_bpe/final.pt \
    --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
    --task ner --train-lang en --eval-lang de --output-dir finetune_out/ner_de
```

**Full eval across many languages, one train pass, no retraining** — the
finetuned head is trained ONCE, then evaluated against every target
language's own test set via `Trainer`'s native multi-dataset support
(`ner`/`pos`/`sib200` only; not a per-language resubmit):

```bash
sbatch jobs/finetune_encoder.sh --checkpoint ... --system bpe --tokenizer-checkpoint ... \
    --task sib200 --train-lang-script eng_Latn --eval-lang-scripts all \
    --output-dir finetune_out/sib200_full
    # --eval-langs / --eval-configs work the same way for ner / pos
    # --max-eval-langs N caps an "all" sweep (prints what got dropped, never silently truncates)
```

Taxi1500 is different: its only HF-hosted dataset ships text with no
labels (`encoder_finetune_taxi1500.py`'s own docstring), so English labels
are auto-downloaded from GitHub instead; non-English needs
`--taxi1500-eval-tsv` pointing at a labeled file you've obtained yourself
(the real Taxi1500-c corpus is gated).

## 4. wandb logging

All three CLIs (`encoder_cli`, `encoder_cli_eval`, `encoder_cli_finetune`)
take `--use-wandb --wandb-project ... --run-name ...`, sharing the same
`pretraining` project as the decoder's own `train.py`/`cli_eval.py` by
default — filter by `job_type` (`encoder_train`, `encoder_eval`,
`finetune`) in the wandb UI to tell them apart.

## 5. Comparing tokenizers: results tooling

`encoder_cli_eval`/`encoder_cli_finetune` can write their result as a
small JSON envelope instead of (or as well as) printing it:

```bash
python3 -m systems.pretraining.encoder_cli_eval ... --label bpe --output results/encoder/pppl_bpe.json
python3 -m systems.pretraining.encoder_cli_finetune ... --label bpe --results-output results/encoder/ner_bpe.json
# ...one file per (tokenizer, benchmark/task) combination...
```

`--label` is what you're comparing BY (defaults to `--system`) — give two
runs of the same tokenizer system distinct labels if they differ in some
other way you want to see side by side (e.g. two fanta checkpoints at
different vocab sizes).

Then combine and render:

```bash
python3 -m scripts.combine_encoder_results \
    --input results/encoder/*.json --output results/encoder_comparison.json

python3 -m scripts.generate_encoder_comparison_table \
    --input results/encoder_comparison.json --output results/encoder_comparison.md
```

The rendered report has two tables: **Summary** (one headline number per
benchmark per tokenizer — for `ner`/`pos`/`taxi1500`/`sib200` this is the
MEAN across every language evaluated, so a single `--eval-lang` run and a
`--eval-langs=all` sweep both collapse to one comparable number), and
**Detailed** (every raw metric key, including the full per-language
breakdown from an `all` sweep). Missing cells render as `--`, never `0`.

## 6. Running the whole suite at once

`jobs/run_encoder_eval_suite.sh` submits all 7 benchmarks/tasks above
(pppl, retrieval, roundtrip, ner, pos, taxi1500, sib200) for every
tokenizer listed in its own `ENCODER_CHECKPOINTS`/`TOKENIZER_SYSTEMS`/
`TOKENIZER_CHECKPOINTS`/`TOKENIZER_VOCAB_JSONS` arrays, then chains
`jobs/combine_encoder_results.sh` to regenerate `results/encoder_comparison.{json,md}`
automatically once everything finishes (`--dependency=afterany`, so one
benchmark failing doesn't block combining what did succeed):

```bash
bash jobs/run_encoder_eval_suite.sh   # run directly on the login node, NOT via sbatch
```

Separate jobs per benchmark, not one chained job — mirrors
`jobs/evaluate_latest_checkpoints.sh`'s own reasoning (one benchmark's own
budget could exceed any single time limit with no way to resume partway).
Edit the four arrays at the top of the script to add more tokenizers; edit
the language-setting variables below them (`RETRIEVAL_PAIR`, `TRAIN_LANG`,
etc.) for different eval languages. Two safety defaults worth knowing:

- **`MAX_EVAL_LANGS=20`** caps the ner/pos/sib200 `--eval-*=all` sweeps —
  WikiANN has 176 configs, Universal Dependencies 350, and finetuning has
  no auto-resubmit (see §3 above), so a genuinely unbounded first run
  risks a wasted multi-hour job that never finishes. Set to `0` once
  you've confirmed a capped run's real wall-clock budget.
- **Roundtrip is skipped automatically** (with a printed warning) if
  `data/bible_nlp` isn't found locally, rather than submitting a job that
  would fail immediately.
