# Evaluating a trained decoder checkpoint

This covers `cli_eval.py` (the downstream benchmark suite) and
`cli_contamination.py` (checking whether that benchmark's own data leaked
into the training corpus). For the separate MLM encoder's own eval suite
(pseudoperplexity/retrieval/roundtrip alignment) and finetuning
(NER/POS/Taxi1500/SIB-200), see **`ENCODER.md`** instead — different
architecture, different eval methodology (zero-shot/likelihood-scoring
here vs. finetuned-head evaluation there), not interchangeable.

Run `python3 -m systems.pretraining.cli_eval --help` /
`cli_contamination.py --help` for the complete flag list; this file covers
the shape of the suite and what each benchmark actually needs.

## 1. Benchmark suite (`cli_eval.py`)

Six benchmarks, all zero-shot/likelihood-scored against
`TransformerLM.generate`/loglikelihood (no finetuning — see
`benchmarks.py`'s own docstring for exact dataset schemas and licensing):

| `--benchmark` | task | needs `--langs` | needs `--lang-pairs` |
|---|---|---|---|
| `xnli` | natural language inference | yes (15 configs) | — |
| `xcopa` | causal commonsense reasoning | yes (11 configs) | — |
| `blimp` | English grammaticality (minimal pairs) | yes (paradigm names, not languages) | — |
| `cola` | English acceptability (MCC-thresholded) | — (English-only) | — |
| `squad` | English extractive QA | — (English-only) | — |
| `flores_mt` | machine translation (BLEU/chrF) | — | yes |

```bash
sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
    --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
    --benchmark xnli --langs en,de,fr,ar,zh --max-examples 1000 \
    --output results/xnli_bpe.json

# --benchmark takes a comma-separated list -- one combined results file, one job
sbatch jobs/evaluate_pretrained.sh --checkpoint checkpoints/pretrain/final.pt \
    --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \
    --benchmark xnli,xcopa,flores_mt --langs en,de,fr --lang-pairs eng:spa,eng:arz \
    --max-examples 500 --output results/all_bpe.json --use-wandb --run-name eval_bpe_50k
```

- Span-family tokenizers need `--vocab-json`, same as everywhere else.
- `--langs` only applies to xnli/xcopa/blimp; `--lang-pairs` only to
  flores_mt; `--max-examples` caps every benchmark (default: full split).
- Single GPU, no DDP — reserved uniformly even though only flores_mt and
  squad (the two that call `TransformerLM.generate()`, unbatched and with
  no KV cache) really need it; xnli/xcopa/blimp/cola are pure
  loglikelihood scoring. Run multiple checkpoints/benchmarks as separate
  submissions rather than one giant job.
- `contamination.py`/`cli_contamination.py` below is the thing to run
  *before* trusting these numbers, not after.

## 2. Contamination check (`cli_contamination.py`)

n-gram overlap between a training corpus and a benchmark's own examples —
a "0% contaminated" result is scoped to whatever you actually scanned, not
a general guarantee.

```bash
python3 -m systems.pretraining.cli_contamination \
    --benchmark xnli --benchmark-langs en,de,fr \
    --corpus-dataset fineweb_edu --corpus-dataset-config sample-10BT \
    --max-corpus-docs 1000000 --output results/contamination_xnli_fineweb.json
```

- No SLURM job wraps this by convention — it's a text-only, no-GPU,
  on-demand scan a user runs and inspects (though `jobs/check_contamination.sh`
  exists for a long/unattended scan; see its own comments for why it has
  **no resume**: a killed run restarts the corpus scan from document 0, so
  size `--max-corpus-docs`/`--time` together so one run actually finishes).
- `--corpus-dataset glot500` needs the same local cache
  `prep_pretraining_data.sh` does (`prepare_glot500.sh` run first) — but
  the benchmark side (xnli/xcopa/flores_mt/blimp/cola/squad) still loads
  live from the HF Hub regardless, so `HF_TOKEN` is still needed even for
  an otherwise fully-local corpus scan.
- `--corpus-langs` uses each corpus's own bare local codes (e.g. glot500's
  `eng`, matching its own cached filenames) — NOT necessarily the same
  code convention `--benchmark-langs` uses for the benchmark side.

## 3. wandb logging

Both share the `pretraining` project with the rest of the decoder
pipeline — `job_type='eval'` / `job_type='contamination_check'`
respectively.
