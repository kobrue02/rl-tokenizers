"""Renders scripts.evaluate_morphology_all's own combined JSON
(results/morphology_all.json / results/morphology_indigenous_panel.json) as
a markdown comparison report (scripts/generate_encoder_comparison_table.py's
own summary/detailed-table convention) plus one TikZ/pgfplots scatter figure
(scripts/generate_tikz_figures.py's own house style -- .dat tables +
standalone/`_body.tex` pairs, no matplotlib) plotting mean MED against mean
Consistency F1, one point per system.

Reuses generate_tikz_figures.py's own family_of/short_name/esc/fam_key/
_FAMILY_COLORS/_FAMILY_RGB/_write_standalone_and_body directly (same family
classification -- fanta is "This work", the other 5 own-trained tokenizers
are "Reproduced baselines", per that module's own _OUR_CONTRIBUTION_NAMES)
rather than re-deriving them, so a tokenizer's color/label here always
matches every other figure in this project. Does NOT reuse
write_scatter_data/gen_landscape_tex/_label_anchor_offsets themselves --
those hardcode "avg_compression"/"spread" as literal column names, which
don't fit mean_med/mean_f1 -- this module has its own small equivalents
instead, same pattern, different columns.

MED vs Consistency F1 pull in OPPOSITE directions with tokenizer
granularity (see this project's own morphology-results analysis,
2026-09-17): MED is unnormalized, so byte/character-level tokenizers
(manta, byt5, canine, blt) rack up a large raw edit distance almost by
construction (many more predicted spans than gold morphemes), while
Consistency F1's substring-overlap check is trivially satisfied more often
at fine granularity. The scatter figure's own xlabel/ylabel spell this out
explicitly (lower MED / higher F1 = "better" on each axis SEPARATELY, but
neither axis alone should be read as "more morphologically aware" without
this caveat) -- don't infer an overall ranking from one axis alone.

--exclude mirrors generate_tikz_figures.py's own flag EXACTLY (same
rationale): Ch.~tokentax's pooled figures must not show fanta before it's
introduced later in the thesis narrative -- pass --exclude fanta for that
scoped view, matching figures/tikz/ vs figures/tikz_with_fanta/'s own split.

This project's own 6 trained tokenizers (bpe/superbpe/fanta/manta/magnet/
parity_bpe) are ALWAYS labeled on the scatter (unlike
generate_tikz_figures.py's gen_landscape_tex, which only labels 3 dynamic
superlatives) -- they're the systems this comparison is actually about;
labeling all ~30 points would be unreadable.

Usage:
    python3 -m scripts.generate_morphology_figures \\
        --input results/morphology_all.json \\
        --indigenous-panel-input results/morphology_indigenous_panel.json \\
        --table-output results/morphology_comparison.md \\
        --output-dir figures/tikz/morphology
"""

import argparse
import json
import os

from common.config_file import parse_args_with_config
from scripts.generate_tikz_figures import (
    _FAMILY_COLORS,
    _FAMILY_RGB,
    _assert_well_formed,
    _write_standalone_and_body,
    esc,
    fam_key,
    family_of,
    short_name,
)

# This project's own 6 trained tokenizers -- always labeled on the scatter,
# regardless of --exclude (an excluded one simply never reaches the row
# list to begin with, so labeling logic never needs to special-case it).
_OWN_SYSTEM_NAMES = {"bpe", "superbpe", "fanta", "manta", "magnet", "parity_bpe"}


def _fertility_by_lang(result):
    """Fertility (common.eval.metrics.fertility, tokens/word -- see that
    function's own docstring) lives at a DIFFERENT path depending on which
    eval_data_source produced `result`: flat at result["fertility"] for a
    normal bouquet/bouquet_test run, but nested under result["combined"]
    for an indigenous_panel run (common.eval.cross_tokenizer.
    evaluate_on_indigenous_panel's own {"combined", "token_parity_by_anchor",
    "morphology_spread", "morphology"} shape -- same distinction
    hf_frontier/evaluate.py's own wandb logging already makes)."""
    if "combined" in result:
        return result["combined"].get("fertility", {})
    return result.get("fertility", {})


def load_morphology_rows(input_path, indigenous_panel_input=None, exclude=None):
    """Returns (rows, all_langs). rows: one dict per system with name/short/
    family/langs ({lang: {med, consistency_f1, n_words, source}})/mean_med/
    mean_f1/mean_fertility/n_langs, sorted by mean_med ascending. A system
    with zero scored languages (e.g. --exclude, or a system present in one
    input but not the other) is dropped entirely -- there's no meaningful
    mean to report.

    mean_fertility is macro-averaged over the SAME language set MED/F1 were
    scored on (not every language that system's own bouquet/bouquet_test run
    covers) -- MorphBPE's own paper reports fertility alongside MED/
    Consistency F1 in one table, and this keeps the three numbers directly
    comparable (same language subset), rather than averaging fertility over
    a much larger, different language set. A language with morphology data
    but no fertility entry (shouldn't happen -- same eval_groups run
    produces both -- but guarded rather than assumed) is skipped for the
    fertility mean only, not dropped from the row entirely.

    indigenous_panel_input: optional second combined JSON (same shape,
    scripts/evaluate_morphology_all.py's own indigenous_panel-sourced
    output) -- its per-system "morphology" languages are UNIONED into the
    same system's languages from --input (aym/cni/shp on top of the 13
    main-panel languages), not treated as a separate row. A system present
    in only one of the two files still gets a row from whichever file has
    it."""
    exclude = set(exclude) if exclude else set()

    def _load(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k != "_failed" and isinstance(v, dict) and k not in exclude}

    combined_langs = {}
    combined_fertility = {}
    for path in [input_path] + ([indigenous_panel_input] if indigenous_panel_input else []):
        for name, result in _load(path).items():
            combined_langs.setdefault(name, {}).update(result.get("morphology", {}))
            combined_fertility.setdefault(name, {}).update(_fertility_by_lang(result))

    rows = []
    all_langs = set()
    for name, langs in combined_langs.items():
        if not langs:
            continue
        meds = [m["med"] for m in langs.values()]
        f1s = [m["consistency_f1"] for m in langs.values()]
        fertilities = [combined_fertility[name][lang] for lang in langs if lang in combined_fertility.get(name, {})]
        all_langs.update(langs)
        rows.append({
            "name": name, "short": short_name(name), "family": family_of(name),
            "langs": langs, "n_langs": len(langs),
            "mean_med": sum(meds) / len(meds), "mean_f1": sum(f1s) / len(f1s),
            "mean_fertility": sum(fertilities) / len(fertilities) if fertilities else None,
        })
    rows.sort(key=lambda r: r["mean_med"])
    return rows, sorted(all_langs)


def _fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def render_markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary_table(rows):
    header = ["system", "family", "n_langs", "mean_med", "mean_consistency_f1", "mean_fertility"]
    out = [
        [r["name"], r["family"], str(r["n_langs"]), _fmt(r["mean_med"]), _fmt(r["mean_f1"]), _fmt(r["mean_fertility"])]
        for r in rows
    ]
    return header, out


def build_detailed_table(rows, all_langs, metric):
    """metric: "med" or "consistency_f1". One row per system, one column
    per language -- "--" for a system/language pair with no gold data or no
    induce_fn coverage for that language (see
    common.eval.morphology.evaluate_morphological_segmentation's own
    "skip, don't error" convention -- this is that same absence, rendered)."""
    header = ["system"] + all_langs
    out = []
    for r in rows:
        row = [r["name"]]
        for lang in all_langs:
            m = r["langs"].get(lang)
            row.append(_fmt(m[metric]) if m else "--")
        out.append(row)
    return header, out


def render_report(rows, all_langs):
    summary_header, summary_rows = build_summary_table(rows)
    med_header, med_rows = build_detailed_table(rows, all_langs, "med")
    f1_header, f1_rows = build_detailed_table(rows, all_langs, "consistency_f1")
    return (
        "## Summary (sorted by mean MED, ascending)\n\n"
        + render_markdown_table(summary_header, summary_rows)
        + "\n\n## Detailed: Morphological Edit Distance per language\n\n"
        + render_markdown_table(med_header, med_rows)
        + "\n\n## Detailed: Morphological Consistency F1 per language\n\n"
        + render_markdown_table(f1_header, f1_rows)
        + "\n"
    )


def compute_families(rows):
    return [f for f in _FAMILY_COLORS if any(r["family"] == f for r in rows)]


def write_scatter_data(rows, families, out_dir):
    for fam in families:
        with open(os.path.join(out_dir, f"scatter_{fam_key(fam)}.dat"), "w", encoding="utf-8") as f:
            f.write("mean_med mean_f1\n")
            for r in rows:
                if r["family"] == fam:
                    f.write(f"{r['mean_med']:.4f} {r['mean_f1']:.4f}\n")


def _label_anchor_offsets(labeled_points, all_x_values, pad_x=0.3, pad_y=0.015):
    """Same "extend away from the nearest edge, alternate vertically"
    convention as generate_tikz_figures.py's own _label_anchor_offsets, just
    parameterized on mean_med/mean_f1 instead of avg_compression/spread."""
    xmin, xmax = min(all_x_values), max(all_x_values)
    xmid = (xmin + xmax) / 2
    by_y = sorted(range(len(labeled_points)), key=lambda i: labeled_points[i]["mean_f1"])
    vertical = {}
    for rank, i in enumerate(by_y):
        vertical[i] = "south" if rank % 2 == 0 else "north"
    out = []
    for i, r in enumerate(labeled_points):
        horiz = "west" if r["mean_med"] < xmid else "east"
        vert = vertical[i]
        dx = pad_x if horiz == "west" else -pad_x
        dy = pad_y if vert == "south" else -pad_y
        out.append((f"{vert} {horiz}", dx, dy))
    return out


def gen_morphology_landscape_tex(rows, families, out_dir, data_prefix=""):
    labeled = [r for r in rows if r["name"] in _OWN_SYSTEM_NAMES]
    anchors = _label_anchor_offsets(labeled, [r["mean_med"] for r in rows]) if labeled else []

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for fam in families:
        r_, g_, b_ = _FAMILY_RGB[fam]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (_FAMILY_COLORS[fam], r_, g_, b_))
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=12cm, height=9cm,",
        r"    xlabel={Mean Morphological Edit Distance (unnormalized; lower = closer to gold morph boundaries)},",
        r"    ylabel={Mean Morphological Consistency F1 (higher = more consistent segmentation)},",
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none},",
        r"    grid=both, grid style={gray!15},",
        r"    axis lines=left,",
        r"    enlarge x limits={abs=0.4}, enlarge y limits={abs=0.02},",
        r"]",
    ]
    for fam in families:
        col = _FAMILY_COLORS[fam]
        body.append(
            r"\addplot+[only marks, mark=*, mark size=2pt, color=%s] table [x=mean_med, y=mean_f1] {%sscatter_%s.dat};"
            % (col, data_prefix, fam_key(fam))
        )
        body.append(r"\addlegendentry{%s}" % fam)
    for r, (anchor, dx, dy) in zip(labeled, anchors):
        body.append(
            r"\node[font=\scriptsize, anchor=%s] at (axis cs:%.4f,%.4f) {%s};"
            % (anchor, r["mean_med"] + dx, r["mean_f1"] + dy, esc(r["short"]))
        )
    body += [r"\end{axis}", r"\end{tikzpicture}"]
    full_tex, _ = _write_standalone_and_body("morphology_landscape", preamble, body, out_dir)
    return full_tex


def generate(input_path, output_dir, indigenous_panel_input=None, data_prefix=None, exclude=None, table_output=None):
    data_prefix = data_prefix if data_prefix is not None else (output_dir.rstrip("/") + "/")
    rows, all_langs = load_morphology_rows(input_path, indigenous_panel_input, exclude=exclude)
    if not rows:
        raise ValueError(f"no system in {input_path!r} has any scored language -- nothing to plot/tabulate")

    os.makedirs(output_dir, exist_ok=True)
    families = compute_families(rows)
    write_scatter_data(rows, families, output_dir)
    tex = gen_morphology_landscape_tex(rows, families, output_dir, data_prefix=data_prefix)
    _assert_well_formed(tex, "fig_morphology_landscape.tex")

    if table_output:
        os.makedirs(os.path.dirname(table_output) or ".", exist_ok=True)
        with open(table_output, "w", encoding="utf-8") as f:
            f.write(render_report(rows, all_langs))
        print(f"wrote comparison tables for {len(rows)} system(s) to {table_output}")
    print(f"wrote morphology_landscape scatter ({len(rows)} systems, {len(all_langs)} languages) to {output_dir}")
    return rows, all_langs


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Render scripts.evaluate_morphology_all's combined JSON as a markdown "
        "comparison report and a TikZ MED-vs-Consistency-F1 scatter figure."
    )
    parser.add_argument("--input", type=str, default="results/morphology_all.json")
    parser.add_argument(
        "--indigenous-panel-input", type=str, default=None,
        help="optional second combined JSON (results/morphology_indigenous_panel.json) -- "
        "its per-system languages (aym/cni/shp) are unioned into the same system's row "
        "from --input, not treated as separate systems",
    )
    parser.add_argument("--output-dir", type=str, default="figures/tikz/morphology")
    parser.add_argument(
        "--data-prefix", type=str, default=None,
        help="path prefix baked into the .dat file references inside the generated .tex "
        "(default: --output-dir itself)",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="comma-separated system name(s) to drop -- e.g. --exclude fanta for "
        "Ch.~tokentax's pooled figures (see generate_tikz_figures.py's own --exclude "
        "for the identical rationale)",
    )
    parser.add_argument("--table-output", type=str, default="results/morphology_comparison.md")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    exclude = [n.strip() for n in args.exclude.split(",")] if args.exclude else None
    generate(
        args.input, args.output_dir, indigenous_panel_input=args.indigenous_panel_input,
        data_prefix=args.data_prefix, exclude=exclude, table_output=args.table_output,
    )


if __name__ == "__main__":
    main()
