"""Figures for the FANTA-vs-other-fairness-aware-tokenizers evaluation
chapter (Ch.~fantaeval): one leaderboard-style bar chart per requested
anchor language, showing each of FANTA/manta/magnet/parity_bpe/flexitokens'
MEAN token parity relative to that anchor across every BOUQuET language --
complementing (not replacing) scripts/generate_scoped_leaderboards.py's own
anchor-invariant "spread" leaderboard for the same 5-tokenizer subset.

DERIVED, not a new evaluation run: each tokenizer's results/
all_tokenizers_comparison.json entry already has token_parity_gm (the
anchor-invariant per-language value common.eval.parity.anchor_invariant_parity
produces), and mean-parity-relative-to-ANY-anchor is exactly
token_parity_gm[lang] / token_parity_gm[anchor] averaged over every lang !=
anchor -- because token_parity_gm[l] = mean_tokens(l) / geomean(mean_tokens(*))
by construction, the ORIGINAL anchor (English) cancels out algebraically, so
this ratio is exact for any anchor, not an approximation. This is valid
specifically because BOUQuET is fully N-way parallel (every language shares
the same underlying content, see Ch.~tokentax's own \\S{sec:tt-setup}) --
contrast common.eval.cross_tokenizer.evaluate_on_indigenous_panel, whose
own docstring explains why it does NOT do this: that panel's sub-corpora are
only pairwise-parallel with their OWN anchor (crk/iu with English,
AmericasNLP with Spanish), so the cancellation this module relies on
wouldn't hold there.

Usage:
    python3 -m scripts.generate_fanta_eval_figures \\
        --input results/all_tokenizers_comparison.json \\
        --anchors eng,spa --output-dir figures/tikz
"""

import argparse
import json
import os
import statistics

from common.eval.parity import _find_anchor_key
from scripts.generate_tikz_figures import (
    compute_families,
    gen_spread_leaderboard_tex,
    short_name,
    write_bar_data,
)

_OUR_WORK = {"fanta"}
_OTHER_APPROACHES = {"manta", "magnet", "parity_bpe", "flexitokens"}
_KEYS = _OUR_WORK | _OTHER_APPROACHES


def _mean_parity_relative_to(models, anchor_lang):
    """{tokenizer_name: mean per-language token_parity_gm[lang]/token_parity_gm[anchor]}
    over every language a tokenizer's token_parity_gm covers, excluding the
    anchor itself (whose ratio is trivially 1.0). Skips (rather than
    fabricates a number for) a tokenizer with no token_parity_gm entry for
    this anchor at all."""
    result = {}
    for name, m in models.items():
        gm = m["token_parity_gm"]
        anchor_key = _find_anchor_key(gm, anchor_lang)
        if anchor_key is None or gm[anchor_key] == 0:
            continue
        anchor_val = gm[anchor_key]
        ratios = [v / anchor_val for lang, v in gm.items() if lang != anchor_key]
        if ratios:
            result[name] = statistics.mean(ratios)
    return result


def build_rows(models, anchor_lang):
    means = _mean_parity_relative_to(models, anchor_lang)
    missing = _KEYS - set(means)
    if missing:
        raise ValueError(
            f"anchor {anchor_lang!r}: no token_parity_gm entry for {sorted(missing)} -- "
            f"can't build the fanta-eval leaderboard without every one of {sorted(_KEYS)}"
        )
    rows = []
    for name in _KEYS:
        rows.append({
            "name": name,
            "short": short_name(name),
            "family": "This work" if name in _OUR_WORK else "Other approaches",
            # named "spread" (not e.g. "parity_vs_anchor") so this reuses
            # gen_spread_leaderboard_tex/write_bar_data completely unchanged
            # -- same convention scripts/generate_tikz_figures.py's own
            # load_indigenous_panel_rows already uses to plug a DIFFERENT
            # metric (fertility_spread) into this same plotting function.
            "spread": means[name],
        })
    rows.sort(key=lambda r: r["spread"])
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def generate_one_anchor(all_results_path, anchor_lang, out_dir, data_prefix=None):
    with open(all_results_path, encoding="utf-8") as f:
        data = json.load(f)
    models = {k: v for k, v in data.items() if k != "_failed" and isinstance(v, dict)}

    rows = build_rows(models, anchor_lang)
    families = compute_families(rows)

    fig_name = f"fanta_eval_parity_vs_{anchor_lang}"
    fig_out_dir = os.path.join(out_dir, fig_name)
    os.makedirs(fig_out_dir, exist_ok=True)
    base_prefix = fig_out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    write_bar_data(rows, families, fig_out_dir)
    tex = gen_spread_leaderboard_tex(
        rows, families, fig_out_dir, data_prefix=base_prefix,
        xlabel=f"Mean token parity relative to {anchor_lang} (derived from anchor-invariant parity)",
        fig_name=fig_name,
    )
    print(f"wrote {fig_name} ({len(rows)} tokenizers) to {fig_out_dir}")
    return tex, rows, families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    parser.add_argument(
        "--anchors", default="eng,spa",
        help="comma-separated anchor language codes -- bare (eng) or full lang_Script "
        "(eng_Latn) both resolve against token_parity_gm's own keys via "
        "common.eval.parity._find_anchor_key",
    )
    parser.add_argument("--output-dir", default="figures/tikz")
    parser.add_argument("--data-prefix", default=None)
    args = parser.parse_args()

    for anchor in args.anchors.split(","):
        generate_one_anchor(args.input, anchor.strip(), args.output_dir, data_prefix=args.data_prefix)


if __name__ == "__main__":
    main()
