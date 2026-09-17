# SLURM job scripts

Grouped by pipeline stage (each subdirectory pairs with the matching
`configs/` subdirectory of the same name -- see `configs/README.md`).
Filenames drop the stage as a redundant prefix (e.g. `train_tokenizer/bpe.sh`,
not `train_tokenizer/train_bpe.sh`) except where stripping it would leave
nothing (`eval/evaluate.sh`, `train_tokenizer/train.sh`) or break the name
(`combine/combine_and_generate_figures.sh`), or where the prefix isn't
actually the stage word (`eval/check_contamination.sh`,
`eval/run_encoder_eval_suite.sh`):

- `train_tokenizer/` -- fitting a `systems/tokenization/*/cli.py` tokenizer
  (`bpe.sh`/`superbpe.sh`/`fanta.sh`/`magnet.sh`/`flexitokens.sh`/`manta.sh`/
  `parity_bpe.sh`), plus `train.sh` (fairtok's own top-level `train.py`
  trainer -- kept unprefixed since there's nothing left to strip)
- `prep/` -- tokenizing a corpus into packed shards
  (`pretraining_data.sh`/`pretraining_data_gpu.sh`), plus the one-time
  local-cache builders (`glot500.sh`, `pile.sh`, `bible_nlp.sh`)
- `pretrain/` -- LM (`pretraining.sh`) / encoder (`encoder_pretraining.sh`)
  pretraining on those shards
- `generate/` -- qualitative sample generation from a pretrained checkpoint
  (`samples.sh`)
- `eval/` -- downstream benchmark eval: this project's own hand-rolled
  harness (`pretrained.sh`, `encoder.sh`, `evaluate.sh`/`blt.sh`/`claude.sh`/
  `hf_frontier.sh`/`own_tokenizers_indigenous_panel.sh` for tokenizer-level
  BOUQuET eval), the EleutherAI/lm-evaluation-harness integration
  (`lm_eval.sh`), plus `check_contamination.sh` and
  `latest_checkpoints.sh`/`run_encoder_eval_suite.sh`
- `finetune/` -- encoder downstream-task finetuning (`encoder.sh`)
- `combine/` -- merging multiple `--output` result files and generating
  comparison figures/tables (`combine_and_generate_figures.sh`,
  `encoder_results.sh`)

Every script accepts `-c config.yml` (see `configs/README.md`) -- copy an
existing `sbatch jobs/<stage>/<script>.sh -c configs/<stage>/<config>.yml`
invocation from a script's own usage comment rather than hand-building a
long flag list each time.

## `train_tokenizer/fanta.sh`'s `RESULT_KEY` env var

`fanta.sh` always auto-submits a post-training eval job on success, and
that job's `--output`/`--result-key` (plus the checkpoint/vocab paths and
wandb run name) are all derived from `RESULT_KEY` (env var, default
`"fanta"`) -- **not** from anything in the `-c config.yml` file. Submitting
an ablation run (`configs/train_tokenizer/fanta_ablation_*_50k.yml`, which
reweight `lambda_fair`/`lambda_rate` to isolate FANTA's two loss terms)
without overriding `RESULT_KEY` would silently overwrite the real fanta
run's own `results/fanta_comparison.json`:

```bash
RESULT_KEY=fanta_ablation_anchor_only sbatch jobs/train_tokenizer/fanta.sh \
    -c configs/train_tokenizer/fanta_ablation_anchor_only_50k.yml
RESULT_KEY=fanta_ablation_gini_only sbatch jobs/train_tokenizer/fanta.sh \
    -c configs/train_tokenizer/fanta_ablation_gini_only_50k.yml
```

Each distinct `RESULT_KEY` gets its own `checkpoints/<key>_<jobid>.pt`,
`results/<key>_comparison.json`, and wandb run name -- see the ablation
configs' own comments for the full motivation and expected outcome.
