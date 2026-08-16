"""Generates thesis-ready TikZ/pgfplots figures from a systems/*/evaluate.py --output
JSON file (e.g. results/hf_frontier_comparison.json, or a combine_eval_results.py output
with Claude's entry merged in). No LaTeX install needed to run this, only to compile
the output.

Five figures (a raw 33-tokenizer x 259-language dump isn't legible in print):
  1. Spread leaderboard: every tokenizer ranked by token_parity_spread (anchor-invariant
     max/min token cost across languages -- see common.eval.parity.anchor_invariant_parity).
  2. Fairness landscape: avg_compression vs spread scatter, one point per tokenizer;
     only the 3 most informative points (best/worst spread, best compression) get labels.
  3. Worst-language heatmap: the only figure that subsets data (259 rows isn't printable).
     Shows the highest max(token_parity) languages across any model. Colors are precomputed
     in Python (ColorBrewer YlOrRd -- pale=parity, deep red=worst, so "worse" reads as
     darker, unlike a directionless colormap like viridis) and baked in as literal \\fill
     commands rather than left to a pgfplots colormap.
  4. Resource-level trend: mean token_parity per tokenizer grouped by Joshi et al. 2020's
     6-level resource taxonomy (see common.data.lang2tax; ~85% of languages resolve against
     it). One thin line per tokenizer, colored by family, full per-language data (not the
     heatmap's worst-20 subset).
  5. Real API cost by provider: 4 subplots (DeepSeek/GPT/Claude/Kimi, see _PROVIDER_PANELS),
     each 6 real Tukey box plots (one per resource level) from each language's own dollar
     cost. Claude renders as a "pending" placeholder until merged into the input. Pricing is
     real, live-fetched (platform.claude.com, developers.openai.com, deepseek.ai; Kimi's
     rate supplied by the user).

Usage:
    python3 generate_tikz_figures.py --input results/hf_frontier_comparison.json --output-dir figures/tikz
    python3 generate_tikz_figures.py -c configs/some_config.yml

No local LaTeX install to compile-test against -- verify with a real compiler
(Overleaf is fine) before trusting the output.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import langcodes
import numpy as np

from common.config_file import parse_args_with_config
from common.data.lang2tax import load_resource_levels

# ColorBrewer YlOrRd-9, pure-python piecewise-linear interpolation (avoids a matplotlib
# dependency). Chosen over a directionless perceptual colormap like viridis, whose
# dark->bright ramp reads backwards for a lower-is-better metric; YlOrRd's pale->deep-red
# matches "worse = darker" and stays colorblind-safe (lightness-only, not hue).
_COST_STOPS = [
    (0.000, (0xff, 0xff, 0xcc)),
    (0.125, (0xff, 0xed, 0xa0)),
    (0.250, (0xfe, 0xd9, 0x76)),
    (0.375, (0xfe, 0xb2, 0x4c)),
    (0.500, (0xfd, 0x8d, 0x3c)),
    (0.625, (0xfc, 0x4e, 0x2a)),
    (0.750, (0xe3, 0x1a, 0x1c)),
    (0.875, (0xbd, 0x00, 0x26)),
    (1.000, (0x80, 0x00, 0x26)),
]

_FAMILY_COLORS = {
    "OpenAI/tiktoken": "openaiCol",
    "Chinese labs": "chinaCol",
    "Meta/Llama": "metaCol",
    "Encoder-only": "encCol",
    "Anthropic": "anthropicCol",
    "This work": "oursCol",
    "Other": "otherCol",
}
_FAMILY_RGB = {
    "OpenAI/tiktoken": (16, 110, 118),
    "Chinese labs": (214, 96, 42),
    "Meta/Llama": (74, 111, 227),
    "Encoder-only": (140, 140, 140),
    "Anthropic": (180, 60, 60),
    "This work": (34, 139, 74),  # distinct green so our own 7 tokenizers stand out from "Other"
    "Other": (163, 79, 168),
}
# Exact system_label strings evaluate.py's TOKENIZERS dict uses for this repo's
# 7 trained tokenizers (combine_eval_results.py keys entries by --result-key,
# which defaults to this) -- exact-match, not the prefix heuristic below.
_REPO_TOKENIZER_NAMES = {"fairtok", "magnet", "flexitokens", "manta", "fanta", "superbpe", "bpe"}
_ENCODER_ONLY_NAMES = {
    "bert-base-cased", "bert-base-multilingual-cased", "distilbert-base-uncased",
    "roberta-base", "xlm-roberta-base", "microsoft/deberta-base",
    "microsoft/deberta-v3-base", "answerdotai/ModernBERT-base", "google/electra-base-discriminator",
}

# Joshi et al. 2020's 6-class names ("The State and Fate of Linguistic
# Diversity and Inclusion in the NLP World"); see common.data.lang2tax for the code->level mapping.
_RESOURCE_LEVEL_LABELS = {
    0: "0: Left-Behinds",
    1: "1: Scraping-Bys",
    2: "2: Hopefuls",
    3: "3: Rising Stars",
    4: "4: Underdogs",
    5: "5: Winners",
}

# Real published input pricing ($/million tokens, cache-miss rate), live-fetched from
# each provider's own pricing page; Kimi-K3's supplied by the user. One panel per
# PROVIDER, not per tokenizer -- GPT-4o represents OpenAI as its current flagship.
# claude-opus-5 renders as a "pending" placeholder (see gen_api_cost_boxplot_tex) until
# its entry is merged into the input file via combine_eval_results.py.
_PROVIDER_PANELS = [
    {"key": "deepseek-ai/DeepSeek-V4-Pro", "display": "DeepSeek V4-Pro", "input": 0.435, "color": (214, 96, 42)},
    {"key": "tiktoken:o200k_base", "display": "GPT-4o", "input": 2.50, "color": (16, 110, 118)},
    {"key": "claude-opus-5", "display": "Claude Opus 5", "input": 5.00, "color": (180, 60, 60)},  # anthropicCol
    {"key": "moonshotai/Kimi-K3", "display": "Kimi K3", "input": 3.00, "color": (60, 150, 130)},
]


def cost_color(t):
    """t=0 (best, at parity) -> pale yellow; t=1 (worst) -> deep red."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_COST_STOPS, _COST_STOPS[1:]):
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(c0[i] + frac * (c1[i] - c0[i])) for i in range(3))
    return _COST_STOPS[-1][1]


def family_of(name):
    if name in _REPO_TOKENIZER_NAMES:
        return "This work"
    lname = name.lower()
    if name.startswith("tiktoken:") or name.startswith("openai"):
        return "OpenAI/tiktoken"
    if "claude" in lname or name.startswith("anthropic"):
        return "Anthropic"
    if name.startswith("Qwen") or name.startswith("deepseek") or name.startswith("moonshotai"):
        return "Chinese labs"
    if name.startswith("meta-llama") or name == "NousResearch/Llama-2-7b-hf":
        return "Meta/Llama"
    if name in _ENCODER_ONLY_NAMES:
        return "Encoder-only"
    return "Other"


def short_name(name):
    return name.split("/")[-1].replace("tiktoken:", "tk:")


def esc(s):
    """LaTeX-safe for use as a literal label (yticklabels, node text, ...)."""
    return s.replace("_", r"\_")


def fam_key(fam):
    """Filesystem/macro-safe stand-in for a family name (used in .dat filenames)."""
    return fam.replace("/", "_").replace(" ", "_")


def lang_display_name(code, _cache={}):
    """'shn_Mymr' -> 'Shan' via langcodes (CLDR-backed), not a hand-maintained table --
    covers the full 259-language BOUQuET set including rare codes, and bare codes with
    no script suffix (e.g. indigenous_panel's "aym"/"crk"/"nah"). Strips ISO 639-3's
    "(individual language)" clarifier. Falls back to the raw code on any lookup failure."""
    if code in _cache:
        return _cache[code]
    name = code
    try:
        if "_" in code:
            lang, script = code.split("_", 1)
            name = langcodes.Language.get(f"{lang}-{script}").language_name()
        else:
            name = langcodes.Language.get(code).language_name()
        name = name.replace(" (individual language)", "")
    except Exception:
        name = code
    _cache[code] = name
    return name


def load_rows(results_path):
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    models = {k: v for k, v in data.items() if k != "_failed" and isinstance(v, dict)}
    missing_spread = [k for k, m in models.items() if "token_parity_spread" not in m]
    if missing_spread:
        plural = len(missing_spread) != 1
        raise ValueError(
            f"{len(missing_spread)} entr{'ies' if plural else 'y'} in {results_path!r} "
            f"{'have' if plural else 'has'} no token_parity_spread -- "
            f"run backfill_anchor_invariant_parity.py on it first: {missing_spread}"
        )

    rows = []
    for name, m in models.items():
        rows.append({
            "name": name,
            "short": short_name(name),
            "family": family_of(name),
            "avg_compression": m["avg_compression"],
            "gini": m["gini"],
            "spread": m["token_parity_spread"],
            "token_parity": m["token_parity"],
        })
    rows.sort(key=lambda r: r["spread"])
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, models


def compute_families(rows):
    return [f for f in _FAMILY_COLORS if any(r["family"] == f for r in rows)]


def write_bar_data(rows, families, out_dir):
    for fam in families:
        with open(os.path.join(out_dir, f"bar_{fam_key(fam)}.dat"), "w", encoding="utf-8") as f:
            f.write("idx spread\n")
            for r in rows:
                if r["family"] == fam:
                    f.write(f"{r['idx']} {r['spread']:.4f}\n")


def write_scatter_data(rows, families, out_dir):
    for fam in families:
        with open(os.path.join(out_dir, f"scatter_{fam_key(fam)}.dat"), "w", encoding="utf-8") as f:
            f.write("avg_compression spread\n")
            for r in rows:
                if r["family"] == fam:
                    f.write(f"{r['avg_compression']:.4f} {r['spread']:.4f}\n")


def _write_standalone_and_body(name, preamble_lines, body_lines, out_dir):
    """Writes fig_<name>.tex (full standalone document, for test-compiles) and
    fig_<name>_body.tex (just the tikzpicture, for \\input-ing into a thesis chapter).
    The body file exists because \\includestandalone needs shell-escape, which many
    Overleaf configs lack -- it then silently renders an empty placeholder instead of
    erroring. \\input-ing the body avoids that (see figures/tikz/README.md)."""
    full_tex = "\n".join(preamble_lines + [r"\begin{document}"] + body_lines + [r"\end{document}"])
    body_tex = "\n".join(body_lines)
    with open(os.path.join(out_dir, f"fig_{name}.tex"), "w", encoding="utf-8") as f:
        f.write(full_tex)
    with open(os.path.join(out_dir, f"fig_{name}_body.tex"), "w", encoding="utf-8") as f:
        f.write(body_tex)
    return full_tex, body_tex


def gen_spread_leaderboard_tex(
    rows, families, out_dir, data_prefix="",
    xlabel="Token-parity spread (max/min across all languages, anchor-invariant)",
    fig_name="spread_leaderboard",
):
    yticklabels = ", ".join(f"{{{esc(r['short'])}}}" for r in rows)
    n = len(rows)
    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    xbar,",
        r"    width=10cm, height=%.1fcm," % max(10.0, n * 0.65),
        r"    xlabel={%s}," % xlabel,
        r"    xmin=0,",
        r"    ytick={%s}," % ",".join(str(r["idx"]) for r in rows),
        r"    yticklabels={%s}," % yticklabels,
        r"    yticklabel style={font=\scriptsize},",
        r"    y dir=reverse,",
        r"    ymin=-0.7, ymax=%d," % (n - 0.3),
        r"    bar width=5pt,",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none, fill=none},",
        r"    axis y line*=left, axis x line*=bottom,",
        r"]",
    ]
    for fam in families:
        col = _FAMILY_COLORS[fam]
        body.append(
            r"\addplot+[xbar, fill=%s, draw=%s, bar shift=0pt] table [x=spread, y=idx] {%sbar_%s.dat};"
            % (col, col, data_prefix, fam_key(fam))
        )
        body.append(r"\addlegendentry{%s}" % fam)
    body += [r"\end{axis}", r"\end{tikzpicture}"]
    full_tex, _ = _write_standalone_and_body(fig_name, preamble, body, out_dir)
    return full_tex


def _label_anchor_offsets(labeled_points, all_x_values, pad=0.06):
    """Picks a two-word TikZ anchor (e.g. "south west") + small (dx, dy) nudge per
    labeled scatter point, so labels extend away from the nearest plot edge and from
    each other -- a fixed anchor=west clips labels near the right edge and collides
    labels close in y. Horizontal side follows which half of the x-range the point is
    in; vertical side alternates in y-sorted order."""
    xmin, xmax = min(all_x_values), max(all_x_values)
    xmid = (xmin + xmax) / 2
    by_y = sorted(range(len(labeled_points)), key=lambda i: labeled_points[i]["spread"])
    vertical = {}
    for rank, i in enumerate(by_y):
        vertical[i] = "south" if rank % 2 == 0 else "north"

    out = []
    for i, r in enumerate(labeled_points):
        horiz = "west" if r["avg_compression"] < xmid else "east"
        vert = vertical[i]
        dx = pad if horiz == "west" else -pad
        dy = pad if vert == "south" else -pad
        out.append((f"{vert} {horiz}", dx, dy))
    return out


def gen_landscape_tex(rows, families, out_dir, data_prefix=""):
    best_spread = min(rows, key=lambda r: r["spread"])
    worst_spread = max(rows, key=lambda r: r["spread"])
    best_compression = max(rows, key=lambda r: r["avg_compression"])
    labeled = [best_spread, worst_spread, best_compression]
    anchors = _label_anchor_offsets(labeled, [r["avg_compression"] for r in rows])

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=12cm, height=9cm,",
        r"    xlabel={Average byte-per-token compression (higher = more compact)},",
        r"    ylabel={Token-parity spread (anchor-invariant; lower = more equitable)},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none},",
        r"    grid=both, grid style={gray!15},",
        r"    axis lines=left,",
        r"    enlarge x limits={abs=0.15}, enlarge y limits={abs=0.6},",
        r"]",
    ]
    for fam in families:
        col = _FAMILY_COLORS[fam]
        body.append(
            r"\addplot+[only marks, mark=*, mark size=2pt, color=%s] table [x=avg_compression, y=spread] {%sscatter_%s.dat};"
            % (col, data_prefix, fam_key(fam))
        )
        body.append(r"\addlegendentry{%s}" % fam)
    for r, (anchor, dx, dy) in zip(labeled, anchors):
        body.append(
            r"\node[font=\scriptsize, anchor=%s] at (axis cs:%.3f,%.3f) {%s};"
            % (anchor, r["avg_compression"] + dx, r["spread"] + dy, esc(r["short"]))
        )
    body += [r"\end{axis}", r"\end{tikzpicture}"]
    full_tex, _ = _write_standalone_and_body("landscape", preamble, body, out_dir)
    return full_tex


def _grouped_positions(rows, families, gap=0.7):
    """Orders models by family (matching the bar/scatter charts), sorted by spread
    within each family, with a `gap`-unit break between family blocks. Axis-agnostic --
    used for the heatmap's row axis. Returns (ordered_rows, {model_name: position},
    [(family, start, end), ...], total_extent)."""
    ordered = []
    positions = {}
    blocks = []
    x = 0.0
    for fam in families:
        fam_rows = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["spread"])
        if not fam_rows:
            continue
        x_start = x
        for r in fam_rows:
            positions[r["name"]] = x
            ordered.append(r)
            x += 1.0
        blocks.append((fam, x_start, x))
        x += gap
    total = x - gap if blocks else 0.0
    return ordered, positions, blocks, total


def _estimate_label_width_cm(s, pt_per_char=3.6):
    """Rough estimate of a \\tiny-font label's rendered width, to reserve enough
    left-margin before the row-label column for the family indicator bar drawn
    further left. Deliberately generous: underestimating lets the bar (drawn on top)
    paint over the longest names; overestimating just leaves harmless whitespace."""
    pt_to_cm = 0.03514
    return len(s) * pt_per_char * pt_to_cm


def gen_heatmap_tex(rows, models, families, out_dir, n_langs=20, n_buckets=40, cell_cm=0.42, gap=0.7):
    """Models go down the rows (family-grouped, same ordering as the leaderboard),
    languages go across the columns (worst first). The opposite orientation (33 models
    across) rendered ~18cm wide -- wider than a normal text block -- forcing an ugly
    \\resizebox shrink or a landscape page; putting the wide axis on the tall dimension
    of a portrait page fits without shrinking."""
    worst_by_lang = {}
    for m in models.values():
        for lang, v in m["token_parity"].items():
            worst_by_lang[lang] = max(worst_by_lang.get(lang, 0), v)
    top_langs = sorted(worst_by_lang, key=lambda l: -worst_by_lang[l])[:n_langs]
    n_cols = len(top_langs)

    ordered, row_pos, blocks, total_height = _grouped_positions(rows, families, gap=gap)
    # .get(l), not [l]: top_langs comes from the UNION of every model's token_parity
    # keys, so a checkpointed in-progress eval may lack some of them -- missing pairs
    # just get no cell below, same skip-don't-crash convention used throughout.
    all_vals = [
        v for r in ordered for l in top_langs
        for v in [models[r["name"]]["token_parity"].get(l)] if v is not None
    ]
    vmax = max(all_vals)

    def bucket_of(v):
        t = math.sqrt(max(0.0, v - 1)) / math.sqrt(max(vmax - 1, 1e-9))
        return min(n_buckets - 1, int(t * n_buckets))

    def y_of(pos):
        """Converts a "rank from top" position into TikZ's bottom-up y."""
        return total_height - pos - 1

    palette = [cost_color(i / (n_buckets - 1)) for i in range(n_buckets)]

    preamble = [r"\documentclass{standalone}", r"\usepackage{tikz}", r"\usepackage{xcolor}"]
    body = [r"\begin{tikzpicture}[x=%.2fcm, y=%.2fcm]" % (cell_cm, cell_cm)]
    for i, (r_, g_, b_) in enumerate(palette):
        body.append(r"\definecolor{heat%d}{RGB}{%d,%d,%d}" % (i, r_, g_, b_))
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB.get(fam, _FAMILY_RGB["Other"])
        body.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS.get(fam, "otherCol"), r_, g_, b_))

    # Cells: no data for this model/language -> left blank, not fabricated.
    for r in ordered:
        y = y_of(row_pos[r["name"]])
        for xi, lang in enumerate(top_langs):
            v = models[r["name"]]["token_parity"].get(lang)
            if v is None:
                continue
            b = bucket_of(v)
            body.append(r"\fill[heat%d] (%d,%.2f) rectangle ++(1,1);" % (b, xi, y))

    # Gridlines drawn per block, not one grid spanning the full height, so the
    # family gap reads as a real visual break rather than a filled seam.
    for _, y0, y1 in blocks:
        body.append(r"\draw[white, line width=0.3pt] (0,%.2f) grid (%d,%.2f);" % (y_of(y1) + 1, n_cols, y_of(y0) + 1))

    # Family header: colored bar + rotated name, left of the row labels. Drawn BEFORE
    # the row labels (so labels win visually) as a backstop against the width estimate
    # (_estimate_label_width_cm) running short -- worst case is a label overlapping a
    # sliver of color, not an invisible label.
    row_label_margin = max((_estimate_label_width_cm(r["short"]) for r in ordered), default=1.0) / cell_cm
    header_x1 = -(0.3 + row_label_margin + 0.5)
    header_x0 = header_x1 - 0.4
    for fam, y0, y1 in blocks:
        col = _FAMILY_COLORS.get(fam, "otherCol")
        body.append(
            r"\fill[%s] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (col, header_x0, y_of(y1) + 1, header_x1, y_of(y0) + 1)
        )
        body.append(
            r"\node[anchor=south, rotate=90, font=\tiny\bfseries] at (%.2f,%.2f) {%s};"
            % (header_x0 - 0.1, (y_of(y1) + 1 + y_of(y0) + 1) / 2, esc(fam))
        )

    # Row labels: plain model short names, left of the grid -- drawn AFTER
    # the family bar (see above) so text always wins visually.
    for r in ordered:
        y = y_of(row_pos[r["name"]])
        body.append(r"\node[anchor=east, font=\tiny] at (-0.2,%.2f) {%s};" % (y + 0.5, esc(r["short"])))

    # Column labels: real language name (via langcodes) + the code in small
    # gray text, rotated, above the grid.
    for xi, lang in enumerate(top_langs):
        body.append(
            r"\node[anchor=south west, rotate=45, font=\tiny] at (%d,%.2f) {%s \textcolor{gray}{\texttt{\tiny(%s)}}};"
            % (xi, total_height + 0.15, esc(lang_display_name(lang)), esc(lang))
        )

    # Legend, to the right of the grid, spanning the full height.
    legend_x0 = n_cols + 0.8
    legend_steps = 20
    for i in range(legend_steps):
        frac = i / (legend_steps - 1)
        b = min(n_buckets - 1, int(frac * n_buckets))
        y0 = total_height * i / legend_steps
        y1 = total_height * (i + 1) / legend_steps
        body.append(r"\fill[heat%d] (%.2f,%.3f) rectangle (%.2f,%.3f);" % (b, legend_x0, y0, legend_x0 + 0.8, y1))
    body.append(r"\draw (%.2f,0) rectangle (%.2f,%.2f);" % (legend_x0, legend_x0 + 0.8, total_height))
    body.append(r"\node[anchor=west, font=\tiny] at (%.2f,0) {1.0$\times$ (parity)};" % (legend_x0 + 0.9))
    body.append(r"\node[anchor=west, font=\tiny] at (%.2f,%.2f) {%.1f$\times$};" % (legend_x0 + 0.9, total_height, vmax))
    body.append(
        r"\node[anchor=west, font=\tiny, align=left, text width=2.2cm] at (%.2f,%.2f) {color scale: "
        r"$\sqrt{v-1}$ (pale = parity, red = worst)};" % (legend_x0 + 0.9, total_height * 0.5)
    )
    body.append(r"\end{tikzpicture}")
    full_tex, _ = _write_standalone_and_body("heatmap", preamble, body, out_dir)
    return full_tex, top_langs


def load_indigenous_panel_rows(results_path):
    """Loads a systems/*/evaluate.py --eval-data-source indigenous_panel --output JSON
    (see evaluate_on_indigenous_panel for the per-model result shape) into the same row
    shape load_rows produces, so compute_families/_grouped_positions work unchanged.
    "spread" = morphology_spread["fertility_spread"], used only to order rows within
    each family -- no figure displays it directly; the mixed-anchor panel is better
    read as two separate anchor-specific token_parity figures instead (see
    gen_indigenous_panel_parity_bars_tex)."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    models = {k: v for k, v in data.items() if k != "_failed" and isinstance(v, dict)}
    missing = [k for k, m in models.items() if "morphology_spread" not in m]
    if missing:
        plural = len(missing) != 1
        raise ValueError(
            f"{len(missing)} entr{'ies' if plural else 'y'} in {results_path!r} "
            f"{'have' if plural else 'has'} no morphology_spread -- expected "
            f"--eval-data-source indigenous_panel results (see "
            f"common.eval.cross_tokenizer.evaluate_on_indigenous_panel): {missing}"
        )

    rows = []
    for name, m in models.items():
        rows.append({
            "name": name,
            "short": short_name(name),
            "family": family_of(name),
            "avg_compression": m["combined"]["avg_compression"],
            "gini": m["combined"]["gini"],
            "spread": m["morphology_spread"]["fertility_spread"],
        })
    rows.sort(key=lambda r: r["spread"])
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows, models


def gen_indigenous_panel_parity_bars_tex(rows, models, families, langs, anchor, fig_name, out_dir, data_prefix=""):
    """One grouped vertical bar chart for this anchor group: languages on the x-axis,
    one clustered bar per tokenizer FAMILY (not per individual tokenizer). Replaces an
    earlier per-language-subplot grid, which was too dense (33+ tokenizers per subplot)
    and overflowed the page at 10 languages; family-level means collapse ~34 bars down
    to len(families).

    Each bar is the MEAN token_parity across models in that family with data for the
    given language -- per-tokenizer detail is available directly from
    results/indigenous_panel_comparison.json. A family with no data for a language
    simply has no bar there.

    The parity=1.0 reference line uses pgfplots' `extra y ticks` + `grid=major`
    rather than a manually-plotted line."""
    family_models = defaultdict(list)
    for r in rows:
        family_models[r["family"]].append(r["name"])

    def family_mean(fam, lang):
        vals = []
        for name in family_models[fam]:
            v = models[name]["token_parity_by_anchor"].get(anchor, {}).get("token_parity", {}).get(lang)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))

    # Families with no data at all for this anchor group are excluded from
    # present_families entirely (not just given an empty bar): pgfplots silently drops
    # a completely-empty-table addplot from its own legend numbering, which shifts
    # every LATER \addlegendentry to mislabel the next real plot. Skipping the
    # \addplot/\addlegendentry pair in Python avoids ever emitting a plot pgfplots
    # would silently drop.
    present_families = [fam for fam in families if any(family_mean(fam, lang) is not None for lang in langs)]
    dropped_families = [fam for fam in families if fam not in present_families]
    if dropped_families:
        print(
            f"  {fig_name}: {len(dropped_families)} family/families with NO data for this anchor group, "
            f"omitted from the legend rather than risk a pgfplots empty-plot misalignment: {dropped_families}"
        )

    for fam in present_families:
        path = os.path.join(out_dir, f"bar_{fig_name}_{fam_key(fam)}.dat")
        with open(path, "w", encoding="utf-8") as f:
            f.write("lang parity\n")
            for lang in langs:
                v = family_mean(fam, lang)
                if v is None:
                    continue
                f.write(f"{lang} {v:.4f}\n")

    symbolic_coords = ",".join(langs)
    xticklabels = ", ".join(f"{{{esc(lang_display_name(lang))}}}" for lang in langs)

    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    ybar, width=%.1fcm, height=6.5cm," % max(10.0, len(langs) * 1.3),
        r"    bar width=%dpt," % max(3, 18 // max(len(families), 1)),
        r"    symbolic x coords={%s}," % symbolic_coords,
        r"    xtick=data,",
        r"    xticklabels={%s}," % xticklabels,
        r"    x tick label style={rotate=40, anchor=east, font=\small},",
        r"    ylabel={mean token parity vs %s}," % esc(anchor),
        r"    ymin=0,",
        r"    enlarge x limits=%.3f," % (0.5 / max(len(langs), 1)),
        r"    extra y ticks={1},",
        r"    extra y tick labels={},",
        r"    extra y tick style={grid=major, major grid style={dashed, gray, thick}},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none, fill=none},",
        r"    axis y line*=left, axis x line*=bottom,",
        r"]",
    ]
    for fam in present_families:
        col_name = _FAMILY_COLORS[fam]
        body.append(
            r"\addplot+[ybar, fill=%s, draw=%s] table [x=lang, y=parity] {%sbar_%s_%s.dat};"
            % (col_name, col_name, data_prefix, fig_name, fam_key(fam))
        )
        body.append(r"\addlegendentry{%s}" % esc(fam))
    body += [r"\end{axis}", r"\end{tikzpicture}"]

    full_tex, _ = _write_standalone_and_body(fig_name, preamble, body, out_dir)
    return full_tex


def generate_indigenous_panel_figures(results_path, out_dir, data_prefix=None):
    """Sibling to generate() (see module docstring) for a --eval-data-source
    indigenous_panel --output JSON instead -- this mixed-anchor panel has no single
    global token_parity, so it needs dedicated loading/figure logic (see
    common.data.indigenous_panel). Two figures, one per anchor language present in
    PAIRS (currently "en": crk/iu, "es": ten AmericasNLP languages)."""
    from common.data.indigenous_panel import PAIRS

    os.makedirs(out_dir, exist_ok=True)
    base_prefix = out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    rows, models = load_indigenous_panel_rows(results_path)
    families = compute_families(rows)

    langs_by_anchor = defaultdict(list)
    for meta in PAIRS.values():
        langs_by_anchor[meta["anchor"]].append(meta["code"])
    for anchor in langs_by_anchor:
        langs_by_anchor[anchor].sort()

    written = {}
    for anchor, langs in sorted(langs_by_anchor.items()):
        anchor_dir = os.path.join(out_dir, f"parity_vs_{anchor}")
        os.makedirs(anchor_dir, exist_ok=True)
        fig_name = f"indigenous_panel_parity_vs_{anchor}"
        tex = gen_indigenous_panel_parity_bars_tex(
            rows, models, families, langs, anchor, fig_name, anchor_dir,
            data_prefix=f"{base_prefix}parity_vs_{anchor}/",
        )
        _assert_well_formed(tex, f"fig_{fig_name}.tex")
        written[anchor] = langs
        print(f"wrote parity_vs_{anchor}/ ({len(langs)} languages, {len(families)} tokenizer families)")

    print(f"{len(rows)} models, {len(families)} families")
    print("wrote one parity_vs_<anchor>/ subdirectory per anchor language under", out_dir)
    return rows, families, written


def gen_resource_level_tex(rows, models, families, out_dir, data_prefix=""):
    """Mean token_parity per tokenizer, grouped by Joshi et al. 2020's 6-level
    resource taxonomy (see common.data.lang2tax; ~85% of languages resolve against it,
    the rest are a genuine gap in that external resource). Uses each model's full
    per-language token_parity dict, not the heatmap's worst-20 subset.

    One thin line per tokenizer, colored by family, `forget plot` so it doesn't spam
    the legend, plus one legend entry per family added manually via `\\addlegendimage`."""
    all_codes = set()
    for m in models.values():
        all_codes.update(m["token_parity"].keys())
    levels, unresolved = load_resource_levels(sorted(all_codes))
    counts = Counter(levels.values())
    present_levels = sorted(counts)

    for r in rows:
        tp = models[r["name"]]["token_parity"]
        by_level = defaultdict(list)
        for lang, lvl in levels.items():
            v = tp.get(lang)
            if v is not None:
                by_level[lvl].append(v)
        path = os.path.join(out_dir, f"resourcelevel_{r['idx']}.dat")
        with open(path, "w", encoding="utf-8") as f:
            f.write("level parity\n")
            for lvl in present_levels:
                vals = by_level.get(lvl)
                if vals:
                    f.write(f"{lvl} {sum(vals) / len(vals):.4f}\n")

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB.get(fam, _FAMILY_RGB["Other"])
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS.get(fam, "otherCol"), r_, g_, b_))

    xticklabels = ", ".join(f"{{{_RESOURCE_LEVEL_LABELS.get(l, str(l))}}}" for l in present_levels)
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=13cm, height=8cm,",
        r"    xlabel={Linguistic resource level (Joshi et al. 2020)},",
        r"    ylabel={Mean token parity vs English},",
        r"    xtick={%s}," % ",".join(str(l) for l in present_levels),
        r"    xticklabels={%s}," % xticklabels,
        r"    x tick label style={font=\scriptsize},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none},",
        r"    grid=both, grid style={gray!15},",
        r"    axis lines=left,",
        r"]",
    ]
    for r in rows:
        col = _FAMILY_COLORS.get(r["family"], "otherCol")
        body.append(
            r"\addplot+[color=%s, mark=*, mark size=1pt, mark options={fill=%s}, line width=0.5pt, forget plot] "
            r"table [x=level, y=parity] {%sresourcelevel_%d.dat};" % (col, col, data_prefix, r["idx"])
        )
    for fam in families:
        col = _FAMILY_COLORS.get(fam, "otherCol")
        body.append(r"\addlegendimage{color=%s, mark=*}" % col)
        body.append(r"\addlegendentry{%s}" % fam)
    body += [r"\end{axis}", r"\end{tikzpicture}"]
    full_tex, _ = _write_standalone_and_body("resource_level", preamble, body, out_dir)
    return full_tex, dict(counts), unresolved


def _panel_title(panel, models):
    """Appends "(N% complete)" whenever a panel's data is meaningfully partial, computed
    from num_total_calls/num_failed_calls (only present on evaluate_claude_on_groups-style
    entries -- a no-op for hf_frontier ones). Makes a checkpointed, still-running eval
    visible directly in the figure, and disappears on its own once the results finish."""
    display = panel["display"]
    m = models.get(panel["key"], {})
    total = m.get("num_total_calls")
    if not total:
        return display
    completed = total - m.get("num_failed_calls", 0)
    pct = 100 * completed / total
    if pct < 99.5:
        return f"{display} ({pct:.0f}\\% complete)"
    return display


def _boxplot_stats(values):
    """Standard Tukey five-number summary: median/quartiles via numpy, whiskers extended
    to the most extreme point within 1.5*IQR (clipped to the real data range), remaining
    points returned separately as outliers for pgfplots' `boxplot prepared`."""
    arr = np.array(sorted(values))
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = arr[(arr >= lo_fence) & (arr <= hi_fence)]
    whisker_lo = float(inliers.min()) if len(inliers) else float(arr.min())
    whisker_hi = float(inliers.max()) if len(inliers) else float(arr.max())
    outliers = arr[(arr < whisker_lo) | (arr > whisker_hi)]
    return {
        "median": float(median), "q1": float(q1), "q3": float(q3),
        "whisker_lo": whisker_lo, "whisker_hi": whisker_hi,
        "outliers": [float(v) for v in outliers],
    }


def gen_api_cost_boxplot_tex(models, out_dir, reference_words=1_000_000):
    """Real per-language cost DISTRIBUTIONS (not just means), one subplot per PROVIDER
    (see _PROVIDER_PANELS), each with 6 Tukey box plots (one per resource level). An
    earlier single-line mean-cost version hid per-language variance by collapsing it
    into one number.

    cost(model, lang) = fertility(model, eng_Latn) * reference_words *
    token_parity(model, lang) * input_price_per_token(model) / 1e6 -- the same
    English-anchored token_parity used elsewhere in this module, priced at the real
    per-token rate. Every resolved language contributes its own point to its level's box.

    Uses plain `minipage`s (not groupplots) for the 2x2 layout. `\\usepgfplotslibrary{statistics}`
    (required for `boxplot prepared`, not loaded by plain pgfplots) is baked into this
    figure's own preamble and called out again in figures/tikz/README.md.

    Claude's panel renders as a "pending" placeholder if "claude-opus-5" isn't in
    `models` yet."""
    present_panels = [p for p in _PROVIDER_PANELS if p["key"] in models]
    all_codes = set()
    for p in present_panels:
        all_codes.update(models[p["key"]]["token_parity"].keys())
    levels, unresolved = load_resource_levels(sorted(all_codes)) if all_codes else ({}, [])
    present_levels = sorted(set(levels.values())) if levels else list(range(6))

    color_names = {p["key"]: f"boxcolor{i}" for i, p in enumerate(_PROVIDER_PANELS)}
    preamble = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{statistics}",
    ]
    for p in _PROVIDER_PANELS:
        r_, g_, b_ = p["color"]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (color_names[p["key"]], r_, g_, b_))

    # Pre-compute every panel's per-level costs up front so one shared ymin/ymax can
    # apply to all 4 -- previously each auto-ranged independently and couldn't be
    # visually compared (e.g. Claude's $10-90 vs DeepSeek's $0.4-6).
    panel_by_level = {}
    all_costs = []
    for panel in present_panels:
        m = models[panel["key"]]
        tp = m["token_parity"]
        eng_tokens = m["fertility"]["eng_Latn"] * reference_words
        by_level = defaultdict(list)
        for lang, lvl in levels.items():
            v = tp.get(lang)
            if v is not None:
                cost = v * eng_tokens * panel["input"] / 1_000_000
                by_level[lvl].append(cost)
                all_costs.append(cost)
        panel_by_level[panel["key"]] = by_level
    if all_costs:
        shared_ymin = 10 ** math.floor(math.log10(min(all_costs)))
        shared_ymax = 10 ** math.ceil(math.log10(max(all_costs)))
    else:
        shared_ymin, shared_ymax = 0.1, 100
    shared_ytick = [10 ** e for e in range(int(math.log10(shared_ymin)), int(math.log10(shared_ymax)) + 1)]
    ytick_opts = [
        r"    log ticks with fixed point,",
        r"    ytick={%s}," % ",".join(str(t) for t in shared_ytick),
    ]

    body = []
    for i, panel in enumerate(_PROVIDER_PANELS):
        col = color_names[panel["key"]]
        # Bottom row (indices 2, 3) gets the shared x-axis label; top row would duplicate it.
        xlabel_opt = [r"    xlabel={Linguistic resource level (Joshi et al. 2020)},"] if i >= 2 else []
        body.append(r"\begin{minipage}[t]{0.48\linewidth}")
        body.append(r"\centering")
        if panel["key"] not in models:
            # Same axis options as the real panels (not axis lines=none) so the
            # placeholder reserves the same layout space and titles stay aligned.
            body += [
                r"\begin{tikzpicture}",
                r"\begin{axis}[",
                r"    width=\linewidth, height=5.2cm,",
                r"    ymode=log, ymin=%g, ymax=%g," % (shared_ymin, shared_ymax),
                r"    xmin=0.5, xmax=6.5,",
                r"    title={%s}," % _panel_title(panel, models),
                r"    title style={font=\small},",
                r"    xtick={1,2,3,4,5,6},",
                r"    xticklabels={0,1,2,3,4,5},",
                r"    x tick label style={font=\tiny},",
                *xlabel_opt,
                r"    ylabel={Cost (USD)},",
                r"    y label style={font=\scriptsize},",
                r"    yticklabel style={font=\tiny},",
                *ytick_opts,
                r"    grid=both, grid style={gray!15},",
                r"]",
                r"\node[align=center, font=\footnotesize] at (axis cs:3.5,3.16) {Pending evaluation\\results};",
                r"\end{axis}",
                r"\end{tikzpicture}",
            ]
        else:
            by_level = panel_by_level[panel["key"]]
            body += [
                r"\begin{tikzpicture}",
                r"\begin{axis}[",
                r"    boxplot/draw direction=y,",
                r"    width=\linewidth, height=5.2cm,",
                r"    ymode=log, ymin=%g, ymax=%g," % (shared_ymin, shared_ymax),
                r"    title={%s}," % _panel_title(panel, models),
                r"    title style={font=\small},",
                r"    xtick={%s}," % ",".join(str(j + 1) for j in range(len(present_levels))),
                r"    xticklabels={%s}," % ",".join(str(l) for l in present_levels),
                r"    x tick label style={font=\tiny},",
                *xlabel_opt,
                r"    ylabel={Cost (USD)},",
                r"    y label style={font=\scriptsize},",
                r"    yticklabel style={font=\tiny},",
                *ytick_opts,
                r"    grid=both, grid style={gray!15},",
                r"]",
            ]
            for lvl in present_levels:
                vals = by_level.get(lvl, [])
                if not vals:
                    continue
                stats = _boxplot_stats(vals)
                outlier_coords = " ".join(f"(0,{v:.6f})" for v in stats["outliers"])
                body.append(
                    r"\addplot+[boxplot prepared={median=%.6f,upper quartile=%.6f,lower quartile=%.6f,"
                    r"upper whisker=%.6f,lower whisker=%.6f},fill=%s,draw=%s,"
                    r"mark options={fill=%s,draw=%s}] coordinates {%s};"
                    % (
                        stats["median"], stats["q3"], stats["q1"], stats["whisker_hi"], stats["whisker_lo"],
                        col, col, col, col, outlier_coords,
                    )
                )
            body += [r"\end{axis}", r"\end{tikzpicture}"]
        body.append(r"\end{minipage}")
        body.append(r"\hfill" if i % 2 == 0 else r"\\[0.5cm]")

    full_tex = "\n".join(preamble + [r"\begin{document}"] + body + [r"\end{document}"])
    body_tex = "\n".join(body)
    with open(os.path.join(out_dir, "fig_api_cost.tex"), "w", encoding="utf-8") as f:
        f.write(full_tex)
    with open(os.path.join(out_dir, "fig_api_cost_body.tex"), "w", encoding="utf-8") as f:
        f.write(body_tex)
    return full_tex, unresolved


def _assert_well_formed(tex, name, expected_tikzpictures=1):
    assert tex.count(r"\begin{document}") == tex.count(r"\end{document}") == 1, f"{name}: unbalanced document"
    assert tex.count(r"\begin{tikzpicture}") == tex.count(r"\end{tikzpicture}") == expected_tikzpictures, (
        f"{name}: expected {expected_tikzpictures} tikzpicture(s)"
    )
    if r"\begin{axis}" in tex:
        assert tex.count(r"\begin{axis}") == tex.count(r"\end{axis}"), f"{name}: unbalanced axis"
    if r"\begin{minipage}" in tex:
        assert tex.count(r"\begin{minipage}") == tex.count(r"\end{minipage}"), f"{name}: unbalanced minipage"
    assert tex.count("{") == tex.count("}"), f"{name}: unbalanced braces"


_FIGURE_SUBDIRS = {
    "spread_leaderboard": "spread_leaderboard",
    "landscape": "landscape",
    "heatmap": "heatmap",
    "resource_level": "resource_level",
    "api_cost": "api_cost",
}


def generate(results_path, out_dir, data_prefix=None):
    """Each figure gets its own subdirectory under out_dir (figures/tikz/spread_leaderboard/,
    etc.) rather than 50+ files flattened together. data_prefix is the base path baked into
    every `table {...}` reference -- needed because \\includestandalone (without shell-escape)
    runs pgfplots from the HOST document's directory, not out_dir, so a bare "bar_X.dat"
    fails once the figure is \\input from elsewhere. Defaults to out_dir itself; override
    if your thesis's include path differs."""
    os.makedirs(out_dir, exist_ok=True)
    base_prefix = out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    def subdir(key):
        path = os.path.join(out_dir, _FIGURE_SUBDIRS[key])
        os.makedirs(path, exist_ok=True)
        return path, f"{base_prefix}{_FIGURE_SUBDIRS[key]}/"

    leaderboard_dir, leaderboard_prefix = subdir("spread_leaderboard")
    landscape_dir, landscape_prefix = subdir("landscape")
    heatmap_dir, _heatmap_prefix = subdir("heatmap")
    resource_level_dir, resource_level_prefix = subdir("resource_level")
    api_cost_dir, _api_cost_prefix = subdir("api_cost")  # no external .dat files -- prefix unused

    rows, models = load_rows(results_path)
    families = compute_families(rows)

    write_bar_data(rows, families, leaderboard_dir)
    write_scatter_data(rows, families, landscape_dir)
    tex1 = gen_spread_leaderboard_tex(rows, families, leaderboard_dir, data_prefix=leaderboard_prefix)
    tex2 = gen_landscape_tex(rows, families, landscape_dir, data_prefix=landscape_prefix)
    tex3, top_langs = gen_heatmap_tex(rows, models, families, heatmap_dir)
    tex4, level_counts, unresolved_langs = gen_resource_level_tex(
        rows, models, families, resource_level_dir, data_prefix=resource_level_prefix
    )
    tex5, _unresolved_cost_langs = gen_api_cost_boxplot_tex(models, api_cost_dir)

    _assert_well_formed(tex1, "fig_spread_leaderboard.tex")
    _assert_well_formed(tex2, "fig_landscape.tex")
    _assert_well_formed(tex3, "fig_heatmap.tex")
    _assert_well_formed(tex4, "fig_resource_level.tex")
    _assert_well_formed(tex5, "fig_api_cost.tex", expected_tikzpictures=4)

    print(f"{len(rows)} models, {len(families)} families, heatmap languages: {top_langs}")
    print(f"resource-level coverage: {sum(level_counts.values())} resolved {dict(sorted(level_counts.items()))}, "
          f"{len(unresolved_langs)} not in Joshi et al.'s taxonomy: {unresolved_langs}")
    print("wrote one subdirectory per figure under", out_dir, "-", ", ".join(_FIGURE_SUBDIRS.values()))
    return rows, families, top_langs


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate TikZ/pgfplots thesis figures from a systems/*/evaluate.py --output JSON file."
    )
    parser.add_argument(
        "--input", type=str, default="results/hf_frontier_comparison.json",
        help="results JSON to read (must already have token_parity_spread -- run "
        "backfill_anchor_invariant_parity.py first if it doesn't)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures/tikz",
        help="directory to write the .tex + .dat files into (created if missing)",
    )
    parser.add_argument(
        "--data-prefix", type=str, default=None,
        help="path prefix baked into the .dat file references inside the generated .tex "
        "(default: --output-dir itself) -- override only if your thesis's main .tex "
        "includes these figures from a different relative location",
    )
    parser.add_argument(
        "--indigenous-panel", action="store_true",
        help="generate the 2-figure common.data.indigenous_panel comparison "
        "(leaderboard + heatmap, see generate_indigenous_panel_figures) instead of the "
        "main 5-figure BOUQuET-based pipeline -- --input must then be a "
        "--eval-data-source indigenous_panel results JSON, not a BOUQuET one",
    )
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    if args.indigenous_panel:
        generate_indigenous_panel_figures(args.input, args.output_dir, data_prefix=args.data_prefix)
    else:
        generate(args.input, args.output_dir, data_prefix=args.data_prefix)


if __name__ == "__main__":
    main()
