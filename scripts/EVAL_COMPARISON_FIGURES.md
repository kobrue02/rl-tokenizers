# Downstream eval comparison figures

Plotting pipeline for comparing tokenizers (e.g. `bpe` vs `fanta`) on
*downstream* pretraining-eval results (XNLI/XCOPA/FLORES-MT for decoders;
NER/POS/SIB-200/Taxi1500/retrieval/roundtrip/pseudoperplexity for encoders).
This is separate from `scripts/generate_tikz_figures.py`, which plots
tokenizer-*intrinsic* metrics (fertility/compression/parity from
`systems/*/evaluate.py`) and never touches these benchmarks at all.

## Changed

- `systems/pretraining/cli_eval.py` — added `--label` (defaults to
  `--system`, mirrors `encoder_cli_eval.py`'s existing convention) and
  wrapped `--output`'s JSON as `{"label", "benchmark", "checkpoint",
  "system", "tokenizer_checkpoint", "results": {<benchmark>: ...}}` instead
  of a bare `{benchmark: result}` dict, so a result file records which
  tokenizer it belongs to.
- `configs/eval_bpe_large.yml`, `eval_fanta_large.yml`, `eval_bpe_50k.yml`,
  `eval_fanta_50k.yml` — added explicit `label:` (`bpe`/`fanta`/`bpe_50k`/
  `fanta_50k`) so the large and 50k runs don't collide if ever compared
  together.
- `systems/pretraining/EVALS.md` — documents the same pipeline from the
  decoder-eval side, cross-reference if editing either.

## New

- `scripts/combine_decoder_results.py` — merges multiple `cli_eval.py
  --output` files into one `{label: {benchmark: result}}` comparison
  (mirrors `scripts/combine_encoder_results.py`).
- `scripts/generate_eval_comparison_figures.py` — grouped bar-chart
  TikZ/pgfplots figures, same house style as `generate_tikz_figures.py` (no
  matplotlib, `.dat` tables + standalone/`_body.tex` pairs). Four figures:
  - `decoder_classification/` — XNLI/XCOPA/BLiMP/SQuAD/CoLA
  - `decoder_flores_mt/` — BLEU/chrF (kept separate: ~[0,100] scale, unlike the others)
  - `encoder_classification/` — retrieval/roundtrip/NER/POS/Taxi1500/SIB-200
  - `encoder_pseudoperplexity/` — kept separate: unbounded, lower-is-better

  A label or category with no data for a given figure is omitted entirely,
  not faked as zero (and not left as an empty bar/axis slot).
- `tests/test_combine_decoder_results.py`,
  `tests/test_generate_eval_comparison_figures.py` — matching the project's
  existing test conventions for the encoder-side equivalents.

## Usage

Once real eval results exist:

```bash
python3 -m scripts.combine_decoder_results \
    --input results/all_bpe_large.json results/all_fanta_large.json \
    --output results/decoder_comparison.json

python3 -m scripts.combine_encoder_results \
    --input results/encoder/*_bpe.json results/encoder/*_fanta.json \
    --output results/encoder_comparison.json

python3 -m scripts.generate_eval_comparison_figures \
    --decoder-input results/decoder_comparison.json \
    --encoder-input results/encoder_comparison.json \
    --output-dir figures/tikz
```

`--decoder-input`/`--encoder-input` are each optional — pass either or both.

## Verification status

Tested against synthetic fixtures only (no real eval results existed yet at
the time this was written): confirmed `combine_decoder_results.py` and
`generate_eval_comparison_figures.py` run end-to-end, produce the expected
`.dat`/`.tex` structure, correctly omit missing labels/categories, and that
`cli_eval.py`'s own smoke test (`run_smoke_test`) still passes after the
`--label`/`results`-wrapper change. **Not** compile-checked against a real
LaTeX install (none available in this environment) — verify with a real
compiler (Overleaf is fine) before trusting the rendered output, same as
`generate_tikz_figures.py`'s own docstring already advises.
