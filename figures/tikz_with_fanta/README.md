# Generated TikZ/pgfplots figures -- WITH fanta included

Sibling of `figures/tikz/README.md`: exact same generator
(`scripts/generate_tikz_figures.py`), exact same 8 figure/table subdirectory
names (`spread_leaderboard/`, `landscape/`, `heatmap/`, `resource_level/`,
`api_cost/`, `tokenizer_summary_table/`, `resource_level_table/`,
`coverage_table/`), same `results/all_tokenizers_comparison.json` input --
the ONLY difference is `fanta` is NOT in `--exclude` here.

This is the Results chapter's figure set: once fanta (this thesis's own
contribution) has been introduced as a method in an earlier chapter, THIS
directory shows where it lands against the full landscape Ch.~tokentax
established without it (`figures/tikz/` -- see that directory's own README
for why fanta is excluded there specifically). `family_of()`'s "This work"
bucket appears here as fanta alone -- see the resource-level-trend figure in
particular, which highlights fanta as the one individually-plotted, bold
star-marker line against every other family's aggregated mean+band.

Regenerate with:

```
python3 -m scripts.generate_tikz_figures \
    --input results/all_tokenizers_comparison.json --output-dir figures/tikz_with_fanta \
    --exclude "Jarbas/m2v-256-bge-reranker-v2-m3,Jarbas/m2v-256-multilingual-e5-small,Yoonyoul/fine-tuned-e5-small-drugproduct,answerdotai/ModernBERT-base,antebe/token_punct_dilute,bert-base-cased,bluexmas/mbart50_ko_vi,distilbert-base-uncased,google/electra-base-discriminator,microsoft/deberta-base,microsoft/deberta-v3-base,roberta-base,slone/mbart-large-51-myv-mul-v1,tiktoken:gpt2,tiktoken:o200k_harmony,tiktoken:p50k_base,tiktoken:p50k_edit,tiktoken:r50k_base,flexitokens" \
    --csv-out results/full_per_language_detail_with_fanta.csv
```

(same exclude list as `figures/tikz/`'s, minus `fanta`.)

For everything else -- layout, embedding in LaTeX, design notes for each
figure/table, the `--data-prefix` mechanics, compiling to test -- see
`figures/tikz/README.md`, which applies identically here (just substitute
`figures/tikz_with_fanta` for `figures/tikz` in every path/command).

A narrower fanta-vs-other-published-methods comparison (fanta vs. manta/
magnet/parity_bpe specifically, not the full landscape) also exists, living
inside `figures/tikz/` itself since it never included fanta's own baseline
"landscape" figures in the first place: `spread_leaderboard_our_work_vs_other_approaches/`,
`fanta_eval_parity_vs_eng/`, `fanta_eval_parity_vs_spa/` (see
`scripts/generate_scoped_leaderboards.py`/`scripts/generate_fanta_eval_figures.py`).
Use whichever comparison fits a given point in the Results chapter -- this
directory for "fanta vs. everyone", those for "fanta vs. the closest
published alternatives" specifically.
