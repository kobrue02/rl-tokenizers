"""Figures for the FANTA-vs-other-fairness-aware-tokenizers evaluation
chapter (Ch.~fantaeval):

  1. generate_one_anchor: one leaderboard-style bar chart per requested
     anchor language, showing each of FANTA/manta/magnet/parity_bpe's MEAN
     token parity relative to that anchor across every BOUQuET language --
     complementing (not replacing) scripts/generate_scoped_leaderboards.py's
     own anchor-invariant "spread" leaderboard for the same 4-tokenizer
     subset.
  2. generate_resource_level_improvement: a DELTA version of Ch.~tokentax's
     fig:resource-level-trend (see generate_tikz_figures.py's
     gen_resource_level_tex) -- instead of each tokenizer's own absolute
     mean token_parity per Joshi resource level, plots fanta's IMPROVEMENT
     over each of the 5 reproduced baselines individually (baseline's mean
     minus fanta's mean, at that level; positive = fanta more equitable).
     Directly visualizes whether fanta's improvement over its own baselines
     is concentrated at low-resource levels (this thesis's core equity
     claim) or flat/negative, which the absolute-value trend line alone
     doesn't make legible.

flexitokens/fairtok excluded from both -- dropped from the project
2026-09-16 (see scripts/generate_tikz_figures.py's _REPO_TOKENIZER_NAMES
comment).

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
    _RESOURCE_LEVEL_LABELS,
    _REPRODUCED_BASELINE_NAMES,
    _resource_levels_and_means,
    _write_standalone_and_body,
    compute_families,
    esc,
    gen_spread_leaderboard_tex,
    load_rows,
    short_name,
    write_bar_data,
)

_OUR_WORK = {"fanta"}
_OTHER_APPROACHES = {"manta", "magnet", "parity_bpe"}
_KEYS = _OUR_WORK | _OTHER_APPROACHES

# Distinct per-baseline color for generate_resource_level_improvement -- the
# main pipeline's single "Reproduced baselines" amber can't distinguish 5
# individual systems from each other, which this figure needs to (unlike
# every other figure, where they're deliberately pooled into one band).
_BASELINE_COLORS = {
    "bpe": "baseBpeCol",
    "superbpe": "baseSuperbpeCol",
    "magnet": "baseMagnetCol",
    "manta": "baseMantaCol",
    "parity_bpe": "baseParityBpeCol",
}
_BASELINE_RGB = {
    "bpe": (31, 119, 180),
    "superbpe": (255, 127, 14),
    "magnet": (44, 160, 44),
    "manta": (214, 39, 40),
    "parity_bpe": (148, 103, 189),
}


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


def generate_resource_level_improvement(all_results_path, out_dir, data_prefix=None):
    """See module docstring (#2). One grouped-bar cluster per Joshi resource
    level, one bar per reproduced baseline within each cluster: baseline's
    own mean token_parity at that level MINUS fanta's, so positive bars mean
    fanta is more equitable than that specific baseline there. Uses the
    SAME English-relative token_parity field (via _resource_levels_and_means)
    that Ch.~tokentax's own fig:resource-level-trend does, not the
    anchor-invariant token_parity_gm generate_one_anchor above uses --
    this figure is meant to read as a delta of THAT specific figure, so it
    needs to share its metric."""
    rows, models = load_rows(all_results_path)
    row_by_name = {r["name"]: r for r in rows}
    missing = ({"fanta"} | _REPRODUCED_BASELINE_NAMES) - set(row_by_name)
    if missing:
        raise ValueError(
            f"{all_results_path!r} is missing expected tokenizer(s): {sorted(missing)}"
        )

    fanta_row = row_by_name["fanta"]
    baseline_names = sorted(_REPRODUCED_BASELINE_NAMES)
    baseline_rows = [row_by_name[name] for name in baseline_names]
    scoped_rows = [fanta_row] + baseline_rows
    _levels, _unresolved, present_levels, _counts, row_level_means = _resource_levels_and_means(
        scoped_rows, models
    )
    fanta_means = row_level_means[fanta_row["idx"]]

    os.makedirs(out_dir, exist_ok=True)
    base_prefix = out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    present_levels_by_baseline = {}
    for name, r in zip(baseline_names, baseline_rows):
        baseline_means = row_level_means[r["idx"]]
        levels_here = [lvl for lvl in present_levels if lvl in fanta_means and lvl in baseline_means]
        present_levels_by_baseline[name] = levels_here
        path = os.path.join(out_dir, f"improvement_{name}.dat")
        with open(path, "w", encoding="utf-8") as f:
            f.write("level improvement\n")
            for lvl in levels_here:
                f.write(f"{lvl} {baseline_means[lvl] - fanta_means[lvl]:.4f}\n")

    present_baselines = [name for name in baseline_names if present_levels_by_baseline[name]]
    dropped = [name for name in baseline_names if name not in present_baselines]
    if dropped:
        print(f"  resource_level_improvement: baseline(s) with no overlapping level data, omitted: {dropped}")

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for name in present_baselines:
        r_, g_, b_ = _BASELINE_RGB[name]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_BASELINE_COLORS[name], r_, g_, b_))

    xticklabels = ", ".join(f"{{{_RESOURCE_LEVEL_LABELS.get(l, str(l))}}}" for l in present_levels)
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    ybar, width=13cm, height=8cm,",
        r"    bar width=%dpt," % max(3, 18 // max(len(present_baselines), 1)),
        r"    symbolic x coords={%s}," % ",".join(str(l) for l in present_levels),
        r"    xtick=data,",
        r"    xticklabels={%s}," % xticklabels,
        r"    x tick label style={font=\scriptsize},",
        r"    xlabel={Linguistic resource level (Joshi et al. 2020)},",
        r"    ylabel={Mean token parity improvement over fanta},",
        r"    enlarge x limits=%.3f," % (0.5 / max(len(present_levels), 1)),
        r"    extra y ticks={0},",
        r"    extra y tick labels={},",
        r"    extra y tick style={grid=major, major grid style={dashed, gray, thick}},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none},",
        r"    grid=both, grid style={gray!15},",
        r"    axis y line*=left, axis x line*=bottom,",
        r"]",
    ]
    for name in present_baselines:
        col = _BASELINE_COLORS[name]
        body.append(
            r"\addplot+[ybar, fill=%s, draw=%s] table [x=level, y=improvement] {%simprovement_%s.dat};"
            % (col, col, base_prefix, name)
        )
        body.append(r"\addlegendentry{%s}" % esc(name))
    body += [r"\end{axis}", r"\end{tikzpicture}"]

    full_tex, _ = _write_standalone_and_body("fanta_resource_level_improvement", preamble, body, out_dir)
    print(f"wrote fanta_resource_level_improvement ({len(present_baselines)} baselines) to {out_dir}")
    return full_tex


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

    improvement_dir = os.path.join(args.output_dir, "fanta_resource_level_improvement")
    generate_resource_level_improvement(
        args.input, improvement_dir,
        data_prefix=f"{args.data_prefix}/fanta_resource_level_improvement" if args.data_prefix else None,
    )


if __name__ == "__main__":
    main()
