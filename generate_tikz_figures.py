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


def gen_spread_leaderboard_tex(rows, families, out_dir, data_prefix=""):
    yticklabels = ", ".join(f"{{{esc(r['short'])}}}" for r in rows)
    n = len(rows)
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
    ]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        lines.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))
    lines += [
        r"\begin{document}",
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
        r"    legend style={at={(0.98,0.02)}, anchor=south east, font=\scriptsize, draw=none, fill=none},",
        r"    axis y line*=left, axis x line*=bottom,",
        r"]",
    ]
    for fam in families:
        col = _FAMILY_COLORS[fam]
        lines.append(
            r"\addplot+[xbar, fill=%s, draw=%s, bar shift=0pt] table [x=spread, y=idx] {%sbar_%s.dat};"
            % (col, col, data_prefix, fam_key(fam))
        )
        lines.append(r"\addlegendentry{%s}" % fam)
    lines += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"]
    tex = "\n".join(lines)
    with open(os.path.join(out_dir, "fig_spread_leaderboard.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    return tex


def gen_landscape_tex(rows, families, out_dir, data_prefix=""):
    best_spread = min(rows, key=lambda r: r["spread"])
    worst_spread = max(rows, key=lambda r: r["spread"])
    best_compression = max(rows, key=lambda r: r["avg_compression"])

    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
    ]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        lines.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))
    lines += [
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=12cm, height=9cm,",
        r"    xlabel={Average byte-per-token compression (higher = more compact)},",
        r"    ylabel={Token-parity spread (anchor-invariant; lower = more equitable)},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none},",
        r"    grid=both, grid style={gray!15},",
        r"    axis lines=left,",
        r"]",
    ]
    for fam in families:
        col = _FAMILY_COLORS[fam]
        lines.append(
            r"\addplot+[only marks, mark=*, mark size=2pt, color=%s] table [x=avg_compression, y=spread] {%sscatter_%s.dat};"
            % (col, data_prefix, fam_key(fam))
        )
        lines.append(r"\addlegendentry{%s}" % fam)
    for r in (best_spread, worst_spread, best_compression):
        lines.append(
            r"\node[font=\scriptsize, anchor=west] at (axis cs:%.3f,%.3f) {%s};"
            % (r["avg_compression"] + 0.05, r["spread"], esc(r["short"]))
        )
    lines += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"]
    tex = "\n".join(lines)
    with open(os.path.join(out_dir, "fig_landscape.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    return tex


def gen_heatmap_tex(rows, models, out_dir, n_langs=20, n_buckets=40, cell_cm=0.42):
    worst_by_lang = {}
    for m in models.values():
        for lang, v in m["token_parity"].items():
            worst_by_lang[lang] = max(worst_by_lang.get(lang, 0), v)
    top_langs = sorted(worst_by_lang, key=lambda l: -worst_by_lang[l])[:n_langs]

    model_order = [r["name"] for r in rows]  # same order as the spread leaderboard
    all_vals = [models[n]["token_parity"][l] for n in model_order for l in top_langs]
    vmax = max(all_vals)

    def bucket_of(v):
        t = math.sqrt(max(0.0, v - 1)) / math.sqrt(max(vmax - 1, 1e-9))
        return min(n_buckets - 1, int(t * n_buckets))

    palette = [viridis(i / (n_buckets - 1)) for i in range(n_buckets)]

    n_cols, n_rows = len(model_order), len(top_langs)
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{tikz}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=%.2fcm, y=%.2fcm]" % (cell_cm, cell_cm),
    ]
    for i, (r_, g_, b_) in enumerate(palette):
        lines.append(r"\definecolor{heat%d}{RGB}{%d,%d,%d}" % (i, r_, g_, b_))

    for yi, lang in enumerate(top_langs):
        for xi, name in enumerate(model_order):
            v = models[name]["token_parity"][lang]
            b = bucket_of(v)
            lines.append(r"\fill[heat%d] (%d,%d) rectangle ++(1,1);" % (b, xi, n_rows - 1 - yi))

    lines.append(r"\draw[white, line width=0.3pt] (0,0) grid (%d,%d);" % (n_cols, n_rows))

    for yi, lang in enumerate(top_langs):
        lines.append(
            r"\node[anchor=east, font=\tiny\ttfamily] at (-0.15,%.1f) {%s};" % (n_rows - 1 - yi + 0.5, esc(lang))
        )
    for xi, r_ in enumerate(rows):
        lines.append(
            r"\node[anchor=north east, rotate=60, font=\tiny] at (%.1f,-0.15) {%s};" % (xi + 0.5, esc(r_["short"]))
        )

    legend_x0 = n_cols + 1.2
    legend_steps = 20
    for i in range(legend_steps):
        frac = i / (legend_steps - 1)
        b = min(n_buckets - 1, int(frac * n_buckets))
        y0 = n_rows * i / legend_steps
        y1 = n_rows * (i + 1) / legend_steps
        lines.append(r"\fill[heat%d] (%.2f,%.3f) rectangle (%.2f,%.3f);" % (b, legend_x0, y0, legend_x0 + 0.8, y1))
    lines.append(r"\draw (%.2f,0) rectangle (%.2f,%.2f);" % (legend_x0, legend_x0 + 0.8, n_rows))
    lines.append(r"\node[anchor=west, font=\tiny] at (%.2f,0) {1.0$\times$ (parity)};" % (legend_x0 + 0.9))
    lines.append(r"\node[anchor=west, font=\tiny] at (%.2f,%.2f) {%.1f$\times$};" % (legend_x0 + 0.9, n_rows, vmax))
    lines.append(
        r"\node[anchor=west, font=\tiny, align=left, text width=2.2cm] at (%.2f,%.2f) {color scale: "
        r"$\sqrt{v-1}$, matching the online dashboard};" % (legend_x0 + 0.9, n_rows * 0.5)
    )
    lines += [r"\end{tikzpicture}", r"\end{document}"]
    tex = "\n".join(lines)
    with open(os.path.join(out_dir, "fig_heatmap.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    return tex, top_langs


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
    tex3, top_langs = gen_heatmap_tex(rows, models, out_dir)

    _assert_well_formed(tex1, "fig_spread_leaderboard.tex")
    _assert_well_formed(tex2, "fig_landscape.tex")
    _assert_well_formed(tex3, "fig_heatmap.tex")

    print(f"{len(rows)} models, {len(families)} families, heatmap languages: {top_langs}")
    print(f"wrote fig_spread_leaderboard.tex, fig_landscape.tex, fig_heatmap.tex to {out_dir}")
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
