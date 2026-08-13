# Experiment configs

Every CLI entry point in this repo (all seven `systems/*/cli.py`, plus
`pretraining/cli.py`, `data_prep.py`, `cli_eval.py`, `cli_generate.py`) now
accepts `-c`/`--config path/to/file.yml` — see `common/config_file.py` for
the full precedence rules. Short version:

- YAML keys are **dest names** (underscores, e.g. `vocab_size`,
  `data_source`, `tokenizer_checkpoint`), matching each command's own
  `--flag` with dashes replaced by underscores.
- A flag passed explicitly on the command line always overrides the same
  key in the YAML file — so `-c configs/train_bpe_50k.yml --vocab-size 999`
  runs with every other value from the file but `vocab_size=999`.
- A required flag (e.g. `cli_eval.py`'s `--checkpoint`/`--system`) can be
  satisfied by the YAML file alone — you don't have to repeat it on the
  command line.
- Repeatable flags (`cli_generate.py`'s `--prompt`) take a YAML **list**
  (`prompt: ["a", "b"]`); every other multi-value flag (`--langs`,
  `--lang-pairs`, `--benchmark`) takes the SAME comma-separated **string**
  form the command line would (`langs: "en,de,fr"`), not a YAML list.

One config file corresponds to ONE pipeline stage's own CLI (its keys are
validated against that stage's own flags, so a file with `--benchmark`'s
`benchmark` key will be rejected by `cli_generate.py`, which has no such
flag) — an experiment spanning multiple stages is multiple files, one per
stage, as in the example below.

## Example: the `bpe_50k` experiment (bpe/superbpe -- CPU-only tokenizer)

```bash
sbatch jobs/train_bpe.sh -c configs/train_bpe_50k.yml
sbatch jobs/prep_pretraining_data.sh -c configs/prep_bpe_50k.yml   # CPU (bpe/superbpe need no GPU)

train_id=$(sbatch --parsable jobs/train_pretraining.sh -c configs/pretrain_bpe_50k.yml)

sbatch --dependency=afterok:$train_id jobs/generate_samples.sh -c configs/generate_bpe_50k.yml
sbatch --dependency=afterok:$train_id jobs/evaluate_pretrained.sh -c configs/eval_bpe_50k.yml
```

## Example: the `fanta_50k` experiment (a NEURAL, span-family tokenizer)

Same shape, two differences: data prep needs a GPU (`jobs/prep_pretraining_data_gpu.sh`,
not the CPU version -- see that script's own docstring for why: a neural
tokenizer's induce_spans is a real forward pass per document), and every
stage past tokenizer training needs `--vocab-json` too (see
`pretraining/tokenizer_adapter.py`'s docstring for why the five neural
systems need it and bpe/superbpe don't).

```bash
sbatch jobs/train_fanta.sh -c configs/train_fanta_50k.yml
sbatch jobs/prep_pretraining_data_gpu.sh -c configs/prep_fanta_50k.yml

train_id=$(sbatch --parsable jobs/train_pretraining.sh -c configs/pretrain_fanta_50k.yml)

sbatch --dependency=afterok:$train_id jobs/generate_samples.sh -c configs/generate_fanta_50k.yml
sbatch --dependency=afterok:$train_id jobs/evaluate_pretrained.sh -c configs/eval_fanta_50k.yml
```

Note `pretrain_fanta_50k.yml` sets its own `output_dir: checkpoints/pretrain_fanta`
-- `pretraining.train`'s default output_dir (`checkpoints/pretrain`) is shared
across every run that doesn't override it, so two pretraining runs active at
the same time (e.g. bpe and fanta) MUST use different `output_dir`s or one
will overwrite the other's checkpoints. Give every new experiment its own
`output_dir` for exactly this reason.

Copy one of these files as a starting point for a new experiment (e.g.
`configs/train_fanta_aggressive_fairness.yml`) rather than hand-building a
long `sbatch ... --flag value --flag value ...` line each time.
