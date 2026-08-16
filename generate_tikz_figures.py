"""Generates thesis-ready TikZ/pgfplots figures directly from a systems/*/evaluate.py
--output JSON file (e.g. results/hf_frontier_comparison.json, or a combine_eval_results.py
output that also has Claude's entry folded in) -- no LaTeX install needed to RUN this, only
to compile what it writes.

Five figures, chosen specifically because a straight 33-tokenizer x 259-language dump is
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
     a hand-picked subset), against every model. Colors are precomputed in Python (ColorBrewer's
     YlOrRd sequential scheme -- pale yellow at parity, deep red at the worst disparity, so
     "worse" reads as a stronger/darker color, unlike a general perceptual colormap such as
     viridis, which has no inherent good/bad direction) and baked in as literal \\fill commands,
     so there's no pgfplots colormap/meshing step to get subtly wrong.
  4. Resource-level trend (fig_resource_level.tex + resourcelevel_*.dat): mean token_parity
     per tokenizer, grouped by Joshi et al. 2020's 6-level linguistic resource taxonomy
     (see common.data.lang2tax for the code->level mapping -- ~85% of this project's
     languages resolve against that external taxonomy; the rest genuinely aren't in it).
     One line per tokenizer (colored by family, no per-tokenizer legend spam), using the
     FULL per-language token_parity data, not the heatmap's worst-20 subset.
  5. Real API cost by provider (fig_api_cost.tex, self-contained -- no external .dat
     needed): 4 subplots (DeepSeek, GPT, Claude, Kimi K3 -- see _PROVIDER_PANELS), each
     with 6 REAL Tukey box-and-whisker plots (one per Joshi et al. resource level) built
     from every resolved language's own dollar cost, not just a group mean. Claude's
     panel renders as a "pending" placeholder until its evaluation results exist and are
     merged in. Uses real, live-fetched pricing from platform.claude.com,
     developers.openai.com, and deepseek.ai (Kimi K3: supplied by the user).

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
from collections import Counter, defaultdict

import langcodes
import numpy as np

from common.config_file import parse_args_with_config
from common.data.lang2tax import load_resource_levels

# ColorBrewer YlOrRd-9 (published, standard sequential scheme), pure-python
# piecewise-linear interpolation -- avoids a matplotlib dependency. Chosen
# over a general perceptual colormap like viridis specifically because
# viridis has no "good/bad" direction: going dark-purple(low)->bright-
# yellow(high) reads BACKWARDS for a lower-is-better cost metric (bright
# intuitively reads as "more/better" to most readers, confirmed as a real
# point of confusion). YlOrRd's pale-yellow(low, near parity)->deep-red
# (high, worse) direction matches the metric's actual meaning, and stays
# colorblind-safe since it varies only in lightness within one hue family,
# not by hue discrimination.
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
    # Distinct green -- every other family is a teal/orange/blue/gray/red/
    # purple, so this project's own 7 trained tokenizers (see
    # _REPO_TOKENIZER_NAMES below) stand out rather than falling into the
    # generic "Other" bucket alongside genuinely unrelated external models.
    "This work": (34, 139, 74),
    "Other": (163, 79, 168),
}
# Exact system_label strings evaluate.py's own TOKENIZERS dict uses for this
# repo's 7 trained tokenizers -- combine_eval_results.py's entries are keyed
# by --result-key, which defaults to exactly this string (see
# common.eval.cross_tokenizer.build_eval_arg_parser's own docstring), so an
# exact-match set is correct here, not a name-prefix heuristic like the
# other families below.
_REPO_TOKENIZER_NAMES = {"fairtok", "magnet", "flexitokens", "manta", "fanta", "superbpe", "bpe"}
_ENCODER_ONLY_NAMES = {
    "bert-base-cased", "bert-base-multilingual-cased", "distilbert-base-uncased",
    "roberta-base", "xlm-roberta-base", "microsoft/deberta-base",
    "microsoft/deberta-v3-base", "answerdotai/ModernBERT-base", "google/electra-base-discriminator",
}

# Joshi et al. 2020's own 6-class names ("The State and Fate of Linguistic
# Diversity and Inclusion in the NLP World") -- see common.data.lang2tax's
# module docstring for the code->level mapping this project uses.
_RESOURCE_LEVEL_LABELS = {
    0: "0: Left-Behinds",
    1: "1: Scraping-Bys",
    2: "2: Hopefuls",
    3: "3: Rising Stars",
    4: "4: Underdogs",
    5: "5: Winners",
}

# Real, published input pricing ($/million tokens): DeepSeek/OpenAI/Anthropic entries
# fetched live from deepseek.ai/pricing, developers.openai.com/api/docs/pricing, and
# platform.claude.com/docs/en/about-claude/pricing; Kimi-K3 pricing supplied directly by
# the user from Moonshot AI's own pricing page. Cache-miss input rates used throughout
# (the standard, non-cached request price) for consistency across providers. One panel
# per PROVIDER (not per tokenizer/encoding) -- OpenAI alone has 4 different tiktoken
# encodings with their own flagship-model prices (GPT-4o/GPT-4-Turbo/davinci-002/
# gpt-3.5-turbo-instruct, see this dict's git history for that earlier line-chart
# version), but a 4-subplot-by-provider layout needs exactly one representative model
# per provider -- GPT-4o chosen as OpenAI's current flagship, per explicit user choice.
# claude-opus-5 is the key evaluate.py's own configs/eval_claude.yml writes results
# under; its panel renders as a "pending" placeholder until that evaluation finishes
# and its entry is merged into the input results file (e.g. via combine_eval_results.py)
# -- gen_api_cost_boxplot_tex checks for the key at generation time, not hardcoded to
# always expect it.
_PROVIDER_PANELS = [
    {"key": "deepseek-ai/DeepSeek-V4-Pro", "display": "DeepSeek V4-Pro", "input": 0.435, "color": (214, 96, 42)},
    {"key": "tiktoken:o200k_base", "display": "GPT-4o", "input": 2.50, "color": (16, 110, 118)},
    {"key": "claude-opus-5", "display": "Claude Opus 5", "input": 5.00, "color": (180, 70, 150)},
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
    """'shn_Mymr' -> 'Shan', 'cmn_Hans' -> 'Mandarin Chinese' -- resolved via
    langcodes (CLDR-backed), NOT a hand-maintained lookup table: confirmed
    live against every one of a real 259-language BOUQuET set with zero
    lookup failures, including rare codes (fia_Copt -> Nobiin, taq_Tfng ->
    Tamasheq, crk_Cans -> Plains Cree). Strips ISO 639-3's "(individual
    language)" macrolanguage-membership clarifier (e.g. on npi/ory/swh/...)
    -- a real, confirmed CLDR annotation, not useful in a chart label.

    Also resolves BARE codes with no script suffix at all (e.g.
    common.data.indigenous_panel's own "aym"/"crk"/"nah" codes, unlike
    BOUQuET's lang_Script stems) -- confirmed live that langcodes resolves
    these directly (aym -> Aymara, crk -> Plains Cree, nah -> "Nahuatl
    languages", a real macrolanguage-umbrella CLDR name, not a lookup
    failure). Falls back to the bare code if it can't be parsed/resolved
    either way (should not happen given the above, but a chart label
    showing the raw code is a safe, honest failure mode, not a crash)."""
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


def _estimate_label_width_cm(s, pt_per_char=3.6):
    """Rough (not exact-metrics) estimate of a \\tiny-font label's rendered
    width, just to reserve enough left-margin for the row-label column
    before placing the family indicator bar further left -- generous by
    design, since underestimating causes the bar (drawn AFTER, i.e. on TOP
    of, the row labels) to visually paint over the longest names, confirmed
    live: a real render at pt_per_char=2.6 clipped exactly the 3 longest
    model names (bert-base-multilingual-cased, electra-base-discriminator,
    distilbert-base-uncased) and nothing else. Overestimating just leaves
    harmless whitespace, so err generous."""
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
    # .get(l) rather than [l]: top_langs is chosen from the UNION of every
    # model's own token_parity keys (see worst_by_lang above), so a language
    # that's one model's worst outlier isn't guaranteed to be present in
    # every OTHER model's own token_parity dict -- confirmed live for a
    # still-in-progress checkpointed eval (e.g. systems/claude_tokenizer's
    # own multi-day run), whose dict only has whichever languages have
    # actually been scored so far, not the full BOUQuET set. A model/lang
    # pair with no data simply contributes nothing here and gets no cell
    # below, same "missing data -> skip, don't crash" convention as every
    # other figure in this module (e.g. gen_indigenous_panel_parity_bars_tex).
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

    # Cells. A model with no data for this language (see all_vals above for
    # why that happens) simply gets no filled cell here -- left blank
    # (the tikzpicture's own background) rather than crashing or
    # fabricating a value.
    for r in ordered:
        y = y_of(row_pos[r["name"]])
        for xi, lang in enumerate(top_langs):
            v = models[r["name"]]["token_parity"].get(lang)
            if v is None:
                continue
            b = bucket_of(v)
            body.append(r"\fill[heat%d] (%d,%.2f) rectangle ++(1,1);" % (b, xi, y))

    # Gridlines drawn PER BLOCK (not one grid spanning the whole height) so
    # the gap between families reads as a real visual break, not a filled seam.
    for _, y0, y1 in blocks:
        body.append(r"\draw[white, line width=0.3pt] (0,%.2f) grid (%d,%.2f);" % (y_of(y1) + 1, n_cols, y_of(y0) + 1))

    # Family header: a colored vertical bar + rotated family name, further
    # left than the row labels -- reserve enough margin (estimated from the
    # longest actual model short-name string) so the bar shouldn't collide
    # with row-label text. Drawn BEFORE the row labels (below in z-order) as
    # a backstop: confirmed live that drawing it AFTER painted over the 3
    # longest model names whose real rendered width came in wider than the
    # estimate -- with labels drawn on top instead, even an imperfect
    # estimate degrades to "label overlaps a sliver of color" rather than
    # "label invisible."
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
    """Loads a systems/*/evaluate.py --eval-data-source indigenous_panel
    --output JSON (see common.eval.cross_tokenizer.evaluate_on_indigenous_panel's
    own docstring for the per-model result shape: {"combined": ...,
    "token_parity_by_anchor": ..., "morphology_spread": ...}) into the SAME
    row shape load_rows produces, so compute_families/_grouped_positions
    (the shared row-ORDERING scaffold every indigenous-panel figure uses,
    for a consistent tokenizer order across figures) work UNCHANGED. "spread"
    here is morphology_spread["fertility_spread"] -- used ONLY to order
    rows within each family (no figure displays this number directly per
    explicit user feedback: the mixed-anchor panel is better read as two
    separate, anchor-specific token_parity figures -- see
    gen_indigenous_panel_parity_bars_tex -- than one anchor-free fertility
    summary; "spread" stays here purely as a stable, meaningful sort key).
    """
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
    """ONE grouped vertical bar chart for this anchor group -- languages
    along the x-axis, one clustered bar per tokenizer FAMILY (not per
    individual tokenizer), colored the same as every other figure
    (_FAMILY_COLORS/_FAMILY_RGB). Replaces an earlier per-language-subplot
    design (a minipage grid, one xbar subplot per language, all 33+
    individual tokenizers shown as separate bars in each) per explicit
    user feedback on a real render: even after fixing that version's own
    width-overflow bug, showing every individual tokenizer once per
    language was too dense to read AND the subplot grid overflowed the
    page at 10 languages. Aggregating to family-level MEANS collapses
    ~34 bars per language down to len(families) (currently 6), and putting
    every language on one shared x-axis instead of one-subplot-per-language
    means the whole comparison is one compact, page-fitting plot instead
    of a tall grid.

    Each family's own bar is the MEAN token_parity across every model in
    that family that has data for a given language -- individual-tokenizer
    detail is deliberately traded away here for a readable overview; a
    reader who wants the per-tokenizer number can still get it from
    results/indigenous_panel_comparison.json directly. A language with NO
    data at all for a given family (e.g. a checkpoint that never covered
    it) simply has no bar there, same graceful-skip convention as every
    other figure in this module.

    A dashed horizontal reference line at parity=1.0 is drawn via pgfplots'
    own `extra y ticks` + `grid=major` mechanism (not a manually-plotted
    line) -- the standard, robust way to add a reference gridline at an
    arbitrary axis value without hand-computing plot-area coordinates.
    """
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

    # Families with NO data at all for this anchor group (e.g. a checkpoint-
    # in-progress model whose relevant phase hasn't produced any completed
    # calls yet) are excluded from `present_families` entirely -- both their
    # .dat file AND their \addplot/\addlegendentry pair are skipped below.
    # This matters beyond just "don't draw an empty bar": pgfplots silently
    # drops a completely-empty-table addplot from its own internal legend
    # numbering, which shifts every LATER \addlegendentry to mislabel the
    # next real plot instead (confirmed live -- a real render showed
    # "Anthropic"'s legend entry colored/positioned as if it were "Other",
    # with "Other"'s own entry missing entirely, because Anthropic's own
    # table was empty at compile time). Skipping the pair in Python avoids
    # ever emitting a plot pgfplots would silently drop.
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
    """Sibling to generate() (see this module's own docstring for the main
    5-figure BOUQuET-based pipeline) for a systems/*/evaluate.py
    --eval-data-source indigenous_panel --output JSON instead -- see
    common.data.indigenous_panel's own module docstring for the panel and
    evaluate_on_indigenous_panel's for why its results need dedicated
    loading/figure logic (this deliberately mixed-anchor panel has no
    single global token_parity the way the BOUQuET comparison does). Two
    figures, one per anchor language actually present in
    common.data.indigenous_panel.PAIRS (currently "en": crk/iu, and "es":
    the ten AmericasNLP languages) -- see
    gen_indigenous_panel_parity_bars_tex's own docstring for the design.
    """
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
    linguistic resource taxonomy (see common.data.lang2tax's own module
    docstring for the code->level mapping and its ~85% coverage of this
    project's languages -- the rest aren't in that external taxonomy at
    all, a real coverage gap in that resource, not something fixable here).
    Uses the FULL per-language token_parity dict each model already has
    (not the heatmap's worst-20 subset), so every resolved language
    contributes to whichever level's mean it belongs to.

    One line per tokenizer (thin, colored by family, `forget plot` so it
    doesn't spam the legend), plus one legend entry per family added
    manually via `\\addlegendimage` -- the same pattern as coloring 33
    tokenizers without a 33-entry legend used elsewhere in this file.
    """
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
    """Appends "(N% complete)" to a panel's title whenever its underlying
    data is meaningfully partial -- self-correcting, since it's computed
    directly from that model's own real num_total_calls/num_failed_calls
    (only present on evaluate_claude_on_groups-style entries; hf_frontier
    entries have neither key, so this is a no-op for them). Confirmed
    useful live: this project's own Claude evaluation is a genuinely
    multi-day job that gets checkpointed and regenerated from partial data
    mid-run -- this marker makes that visible directly IN the figure,
    not just in a commit message or chat note that's easy to forget, and
    disappears on its own once a regenerate uses the finished results
    (no manual edit to remember/revert)."""
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
    """Standard Tukey five-number summary from REAL data (not simulated):
    median/quartiles via numpy, whiskers extended to the most extreme point
    within 1.5*IQR of the quartiles (clipped to the actual data range, never
    invented beyond it), remaining points beyond that returned separately as
    outliers -- pgfplots' `boxplot prepared` plots these as individual
    points, same as any standard box-and-whisker plot."""
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
    """Real per-language cost DISTRIBUTIONS (not just means), one subplot per
    PROVIDER (DeepSeek, GPT, Claude, Kimi K3 -- see _PROVIDER_PANELS), each
    with 6 real Tukey box-and-whisker plots (one per Joshi et al. resource
    level). Confirmed live that an earlier single-chart line version (mean
    cost only) visually tangled two similarly-priced series together and hid
    all per-language variance -- a box plot surfaces that variance directly
    instead of collapsing it into one number.

    cost(model, lang) = fertility(model, eng_Latn) * reference_words *
    token_parity(model, lang) * input_price_per_token(model) / 1e6 -- the
    SAME English-anchored token_parity this whole module already uses,
    converted to an absolute token count via that model's own measured
    English fertility, then priced at its real per-token rate. Every
    resolved language contributes its OWN point to its level's box, not
    just a group mean.

    Uses plain LaTeX `minipage`s (not pgfplots' groupplots library) for the
    2x2 layout -- lower-risk than a library this project hasn't used
    elsewhere, given no local LaTeX install to compile-test against.
    `\\usepgfplotslibrary{statistics}` is required for `boxplot prepared`
    and is NOT loaded by plain `\\usepackage{pgfplots}` alone -- baked into
    this figure's own preamble, and called out again in figures/tikz/README.md
    since the user must also add it to their thesis's main preamble.

    Claude's panel renders as a "pending" placeholder if "claude-opus-5"
    (the key configs/eval_claude.yml's own evaluate.py run writes results
    under) isn't in `models` yet -- checked at generation time, not
    hardcoded to always expect it.
    """
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

    body = []
    for i, panel in enumerate(_PROVIDER_PANELS):
        col = color_names[panel["key"]]
        body.append(r"\begin{minipage}[t]{0.48\linewidth}")
        body.append(r"\centering")
        if panel["key"] not in models:
            # Same axis options as the real panels below (grid, ticks, ylabel,
            # log-scale range) -- confirmed live that an empty `axis lines=none`
            # placeholder does NOT reserve the same layout space pgfplots gives
            # a real, populated axis, so the title/text ended up misaligned
            # against the other 3 panels. Matching the structure (just with no
            # \addplot data) keeps every panel's title at the same height.
            body += [
                r"\begin{tikzpicture}",
                r"\begin{axis}[",
                r"    width=\linewidth, height=5.2cm,",
                r"    ymode=log, ymin=0.1, ymax=100,",
                r"    xmin=0.5, xmax=6.5,",
                r"    title={%s}," % _panel_title(panel, models),
                r"    title style={font=\small},",
                r"    xtick={1,2,3,4,5,6},",
                r"    xticklabels={0,1,2,3,4,5},",
                r"    x tick label style={font=\tiny},",
                r"    ylabel={Cost (USD)},",
                r"    y label style={font=\scriptsize},",
                r"    yticklabel style={font=\tiny},",
                r"    grid=both, grid style={gray!15},",
                r"]",
                r"\node[align=center, font=\footnotesize] at (axis cs:3.5,3.16) {Pending evaluation\\results};",
                r"\end{axis}",
                r"\end{tikzpicture}",
            ]
        else:
            m = models[panel["key"]]
            tp = m["token_parity"]
            eng_tokens = m["fertility"]["eng_Latn"] * reference_words
            by_level = defaultdict(list)
            for lang, lvl in levels.items():
                v = tp.get(lang)
                if v is not None:
                    by_level[lvl].append(v * eng_tokens * panel["input"] / 1_000_000)
            body += [
                r"\begin{tikzpicture}",
                r"\begin{axis}[",
                r"    boxplot/draw direction=y,",
                r"    width=\linewidth, height=5.2cm,",
                r"    ymode=log,",
                r"    title={%s}," % _panel_title(panel, models),
                r"    title style={font=\small},",
                r"    xtick={%s}," % ",".join(str(j + 1) for j in range(len(present_levels))),
                r"    xticklabels={%s}," % ",".join(str(l) for l in present_levels),
                r"    x tick label style={font=\tiny},",
                r"    ylabel={Cost (USD)},",
                r"    y label style={font=\scriptsize},",
                r"    yticklabel style={font=\tiny},",
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
    figures/tikz/landscape/, etc.) -- one figure's .tex + .dat files sitting together,
    rather than 50+ files flattened into one directory. data_prefix: BASE path prefix
    (before the per-figure subdirectory name gets appended) baked into every
    `table {...}` reference inside fig_spread_leaderboard.tex/fig_landscape.tex/
    fig_resource_level.tex/fig_api_cost.tex, e.g. "figures/tikz". Needed because
    \\includestandalone (without shell-escape) runs pgfplots from the HOST document's
    own directory, not from out_dir -- a bare filename like "bar_X.dat" only resolves
    when compiling standalone directly inside that figure's own subdirectory, and
    fails with "Could not read table file" once the figure is included from a
    thesis's main .tex elsewhere. Defaults to out_dir itself (normalized to forward
    slashes), which is correct whenever the main document compiles from the same
    root this script was run from -- override if your actual include path differs.
    """
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
    tex5, unresolved_cost_langs = gen_api_cost_boxplot_tex(models, api_cost_dir)

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
