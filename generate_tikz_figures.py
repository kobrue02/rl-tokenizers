"""Generates thesis-ready TikZ/pgfplots figures directly from a systems/*/evaluate.py
--output JSON file (e.g. results/hf_frontier_comparison.json, or a combine_eval_results.py
output that also has Claude's entry folded in) -- no LaTeX install needed to RUN this, only
to compile what it writes.

Three figures, chosen specifically because a straight 33-tokenizer x 259-language dump is
unreadable in print:

  1. Spread leaderboard (fig_spread_leaderboard.tex + bar_*.dat): every tokenizer in the
     input file, ranked by token_parity_spread (max/min token cost across every language --
     anchor-invariant, see common.eval.parity.anchor_invariant_parity's own docstring for
     why that matters). No reduction needed: spread already summarizes the FULL language
     set for each model.
  2. Fairness landscape (fig_landscape.tex + scatter_*.dat): avg_compression vs spread,
     every tokenizer as one point. Also no reduction -- only the 3 most informative points
     get text labels (best/worst spread, best compression) instead of 33 overlapping labels.
  3. Worst-language heatmap (fig_heatmap.tex, self-contained -- no external .dat needed):
     the ONE figure that reduces anything, because a 259-row table isn't legible in print.
     Shows the languages with the highest max(token_parity) across ANY model in the input
     (the same data-driven rule the interactive dashboard's heatmap tab already uses -- not
     a hand-picked subset), against every model. Colors are precomputed in Python (a viridis
     approximation, colorblind-safe and reasonable in grayscale) and baked in as literal
     \\fill commands, so there's no pgfplots colormap/meshing step to get subtly wrong.

Usage:
    python3 generate_tikz_figures.py --input results/hf_frontier_comparison.json --output-dir figures/tikz
    python3 generate_tikz_figures.py -c configs/some_config.yml

No LaTeX was available to compile-test these when this module was written --
verify with a real compiler (Overleaf is fine) before trusting the output.
"""

import argparse
import json
import math
import os

import langcodes

from common.config_file import parse_args_with_config

# Viridis approximation (published anchor stops), pure-python piecewise-linear
# interpolation -- avoids a matplotlib dependency just to pick print-safe,
# colorblind-safe colors.
_VIRIDIS_STOPS = [
    (0.00, (0x44, 0x01, 0x54)),
    (0.13, (0x48, 0x28, 0x78)),
    (0.25, (0x3e, 0x49, 0x89)),
    (0.38, (0x31, 0x68, 0x8e)),
    (0.50, (0x26, 0x82, 0x8e)),
    (0.63, (0x1f, 0x9e, 0x89)),
    (0.75, (0x35, 0xb7, 0x79)),
    (0.88, (0x6e, 0xce, 0x58)),
    (1.00, (0xfd, 0xe7, 0x25)),
]

_FAMILY_COLORS = {
    "OpenAI/tiktoken": "openaiCol",
    "Chinese labs": "chinaCol",
    "Meta/Llama": "metaCol",
    "Encoder-only": "encCol",
    "Anthropic": "anthropicCol",
    "Other": "otherCol",
}
_FAMILY_RGB = {
    "OpenAI/tiktoken": (16, 110, 118),
    "Chinese labs": (214, 96, 42),
    "Meta/Llama": (74, 111, 227),
    "Encoder-only": (140, 140, 140),
    "Anthropic": (180, 60, 60),
    "Other": (163, 79, 168),
}
_ENCODER_ONLY_NAMES = {
    "bert-base-cased", "bert-base-multilingual-cased", "distilbert-base-uncased",
    "roberta-base", "xlm-roberta-base", "microsoft/deberta-base",
    "microsoft/deberta-v3-base", "answerdotai/ModernBERT-base", "google/electra-base-discriminator",
}


def viridis(t):
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_VIRIDIS_STOPS, _VIRIDIS_STOPS[1:]):
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(round(c0[i] + frac * (c1[i] - c0[i])) for i in range(3))
    return _VIRIDIS_STOPS[-1][1]


def family_of(name):
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
    """'shn_Mymr' -> 'Shan', 'cmn_Hans' -> 'Mandarin Chinese' -- resolved via
    langcodes (CLDR-backed), NOT a hand-maintained lookup table: confirmed
    live against every one of a real 259-language BOUQuET set with zero
    lookup failures, including rare codes (fia_Copt -> Nobiin, taq_Tfng ->
    Tamasheq, crk_Cans -> Plains Cree). Strips ISO 639-3's "(individual
    language)" macrolanguage-membership clarifier (e.g. on npi/ory/swh/...)
    -- a real, confirmed CLDR annotation, not useful in a chart label. Falls
    back to the bare code if it can't be parsed/resolved (should not happen
    given the above, but a chart label showing the raw code is a safe,
    honest failure mode, not a crash)."""
    if code in _cache:
        return _cache[code]
    name = code
    if "_" in code:
        lang, script = code.split("_", 1)
        try:
            name = langcodes.Language.get(f"{lang}-{script}").language_name()
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


def write_bar_and_scatter_data(rows, out_dir):
    families = [f for f in _FAMILY_COLORS if any(r["family"] == f for r in rows)]
    for fam in families:
        with open(os.path.join(out_dir, f"bar_{fam_key(fam)}.dat"), "w", encoding="utf-8") as f:
            f.write("idx spread\n")
            for r in rows:
                if r["family"] == fam:
                    f.write(f"{r['idx']} {r['spread']:.4f}\n")
        with open(os.path.join(out_dir, f"scatter_{fam_key(fam)}.dat"), "w", encoding="utf-8") as f:
            f.write("avg_compression spread\n")
            for r in rows:
                if r["family"] == fam:
                    f.write(f"{r['avg_compression']:.4f} {r['spread']:.4f}\n")
    return families


def _write_standalone_and_body(name, preamble_lines, body_lines, out_dir):
    """Writes BOTH fig_<name>.tex (a full \\documentclass{standalone} document,
    for standalone test-compiles) and fig_<name>_body.tex (just body_lines --
    the \\begin{tikzpicture}...\\end{tikzpicture} content, nothing else) for
    \\input-ing directly into a thesis chapter. The _body.tex file exists
    because \\includestandalone requires shell-escape (to recompile the
    referenced file into its own PDF) -- when that's unavailable (the
    default on many Overleaf compiler configs), it silently falls back to an
    empty "file not found" placeholder box instead of erroring loudly.
    \\input-ing the body directly sidesteps that entirely: it's plain TikZ/
    pgfplots code compiled in the SAME pass as the rest of the thesis, so it
    only needs the relevant packages/colors declared once in the main
    preamble (see this repo's figures/tikz/README.md)."""
    full_tex = "\n".join(preamble_lines + [r"\begin{document}"] + body_lines + [r"\end{document}"])
    body_tex = "\n".join(body_lines)
    with open(os.path.join(out_dir, f"fig_{name}.tex"), "w", encoding="utf-8") as f:
        f.write(full_tex)
    with open(os.path.join(out_dir, f"fig_{name}_body.tex"), "w", encoding="utf-8") as f:
        f.write(body_tex)
    return full_tex, body_tex


def gen_spread_leaderboard_tex(rows, families, out_dir, data_prefix=""):
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
        r"    xlabel={Token-parity spread (max/min across all languages, anchor-invariant)},",
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
    full_tex, _ = _write_standalone_and_body("spread_leaderboard", preamble, body, out_dir)
    return full_tex


def _label_anchor_offsets(labeled_points, all_x_values, pad=0.06):
    """Picks a two-word TikZ anchor (e.g. "south west") + small (dx, dy) nudge
    for each of a handful of annotated scatter points, so the label text
    extends AWAY from both the nearest plot edge and from any other labeled
    point instead of always extending the same direction -- confirmed live
    (real Overleaf render) that a fixed anchor=west with a small +x offset
    clips or fully hides labels for points near the plot's right edge (text
    extends rightward straight into/past the boundary), and that two labels
    whose points are close in y collide with each other since both were
    nudged the same direction. Horizontal side (west/east) is chosen by
    which half of the overall x-range the point falls in (so a label never
    extends toward the nearer edge); vertical side (north/south) alternates
    across the points in y-sorted order (so any two vertically-close points
    end up on opposite sides of their own markers)."""
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
    """Orders models by family (matching the bar/scatter charts' grouping),
    sorted by spread within each family, with a `gap`-unit break between
    family blocks. Axis-agnostic -- used for the heatmap's ROW axis (models
    now go down the page, not across it -- see gen_heatmap_tex's own
    docstring for why). Returns (ordered_rows, {model_name: position},
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


def _estimate_label_width_cm(s, pt_per_char=2.6):
    """Rough (not exact-metrics) estimate of a \\tiny-font label's rendered
    width, just to reserve enough left-margin for the row-label column
    before placing the family indicator bar further left -- generous by
    design, since underestimating causes real overlap and overestimating
    just leaves harmless whitespace."""
    pt_to_cm = 0.03514
    return len(s) * pt_per_char * pt_to_cm


def gen_heatmap_tex(rows, models, families, out_dir, n_langs=20, n_buckets=40, cell_cm=0.42, gap=0.7):
    """Models go down the ROWS (y-axis, family-grouped, same as the
    leaderboard's ordering), languages go across the COLUMNS (x-axis, plain
    left-to-right, worst first) -- confirmed live (a real Overleaf render)
    that the opposite orientation (33 models across, 20 languages down) came
    out ~18cm wide, wider than a normal page's text width, forcing either an
    ugly \\resizebox shrink (which shrinks the already-\\tiny labels into
    illegibility) or a landscape page. Putting the WIDE axis (33 models) on
    the TALL dimension of a portrait page and the NARROW axis (20 languages)
    on its width fits comfortably without shrinking anything."""
    worst_by_lang = {}
    for m in models.values():
        for lang, v in m["token_parity"].items():
            worst_by_lang[lang] = max(worst_by_lang.get(lang, 0), v)
    top_langs = sorted(worst_by_lang, key=lambda l: -worst_by_lang[l])[:n_langs]
    n_cols = len(top_langs)

    ordered, row_pos, blocks, total_height = _grouped_positions(rows, families, gap=gap)
    all_vals = [models[r["name"]]["token_parity"][l] for r in ordered for l in top_langs]
    vmax = max(all_vals)

    def bucket_of(v):
        t = math.sqrt(max(0.0, v - 1)) / math.sqrt(max(vmax - 1, 1e-9))
        return min(n_buckets - 1, int(t * n_buckets))

    def y_of(pos):
        """Converts a "rank from top" position into TikZ's bottom-up y."""
        return total_height - pos - 1

    palette = [viridis(i / (n_buckets - 1)) for i in range(n_buckets)]

    preamble = [r"\documentclass{standalone}", r"\usepackage{tikz}", r"\usepackage{xcolor}"]
    body = [r"\begin{tikzpicture}[x=%.2fcm, y=%.2fcm]" % (cell_cm, cell_cm)]
    for i, (r_, g_, b_) in enumerate(palette):
        body.append(r"\definecolor{heat%d}{RGB}{%d,%d,%d}" % (i, r_, g_, b_))
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB.get(fam, _FAMILY_RGB["Other"])
        body.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS.get(fam, "otherCol"), r_, g_, b_))

    # Cells.
    for r in ordered:
        y = y_of(row_pos[r["name"]])
        for xi, lang in enumerate(top_langs):
            v = models[r["name"]]["token_parity"][lang]
            b = bucket_of(v)
            body.append(r"\fill[heat%d] (%d,%.2f) rectangle ++(1,1);" % (b, xi, y))

    # Gridlines drawn PER BLOCK (not one grid spanning the whole height) so
    # the gap between families reads as a real visual break, not a filled seam.
    for _, y0, y1 in blocks:
        body.append(r"\draw[white, line width=0.3pt] (0,%.2f) grid (%d,%.2f);" % (y_of(y1) + 1, n_cols, y_of(y0) + 1))

    # Row labels: plain model short names, left of the grid.
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

    # Family header: a colored vertical bar + rotated family name, further
    # left than the row labels -- reserve enough margin (estimated from the
    # longest actual model short-name string) so the bar never collides with
    # row-label text.
    row_label_margin = max((_estimate_label_width_cm(r["short"]) for r in ordered), default=1.0) / cell_cm
    header_x1 = -(0.3 + row_label_margin + 0.3)
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
        r"$\sqrt{v-1}$, matching the online dashboard};" % (legend_x0 + 0.9, total_height * 0.5)
    )
    body.append(r"\end{tikzpicture}")
    full_tex, _ = _write_standalone_and_body("heatmap", preamble, body, out_dir)
    return full_tex, top_langs


def _assert_well_formed(tex, name):
    for env in ("document", "tikzpicture"):
        assert tex.count(rf"\begin{{{env}}}") == tex.count(rf"\end{{{env}}}") == 1, f"{name}: unbalanced {env}"
    if r"\begin{axis}" in tex:
        assert tex.count(r"\begin{axis}") == tex.count(r"\end{axis}"), f"{name}: unbalanced axis"
    assert tex.count("{") == tex.count("}"), f"{name}: unbalanced braces"


def generate(results_path, out_dir, data_prefix=None):
    """data_prefix: path prefix baked into every `table {...}` reference inside
    fig_spread_leaderboard.tex/fig_landscape.tex, e.g. "figures/tikz/". Needed
    because \\includestandalone (without shell-escape) runs pgfplots from the
    HOST document's own directory, not from out_dir -- a bare filename like
    "bar_X.dat" only resolves when compiling standalone directly inside
    out_dir, and fails with "Could not read table file" once the figure is
    included from a thesis's main .tex elsewhere. Defaults to out_dir itself
    (normalized to forward slashes, trailing slash added), which is correct
    whenever the main document compiles from the same root this script was
    run from -- override if your actual include path differs (e.g. the
    figures live one level up from where the main .tex compiles).
    """
    os.makedirs(out_dir, exist_ok=True)
    if data_prefix is None:
        data_prefix = out_dir.replace(os.sep, "/")
        if data_prefix and not data_prefix.endswith("/"):
            data_prefix += "/"
    rows, models = load_rows(results_path)

    families = write_bar_and_scatter_data(rows, out_dir)
    tex1 = gen_spread_leaderboard_tex(rows, families, out_dir, data_prefix=data_prefix)
    tex2 = gen_landscape_tex(rows, families, out_dir, data_prefix=data_prefix)
    tex3, top_langs = gen_heatmap_tex(rows, models, families, out_dir)

    _assert_well_formed(tex1, "fig_spread_leaderboard.tex")
    _assert_well_formed(tex2, "fig_landscape.tex")
    _assert_well_formed(tex3, "fig_heatmap.tex")

    print(f"{len(rows)} models, {len(families)} families, heatmap languages: {top_langs}")
    print(f"wrote fig_{{spread_leaderboard,landscape,heatmap}}.tex (standalone, for test-compiling)")
    print(f"wrote fig_{{spread_leaderboard,landscape,heatmap}}_body.tex (for \\input from your thesis) to {out_dir}")
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
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    generate(args.input, args.output_dir, data_prefix=args.data_prefix)


if __name__ == "__main__":
    main()
