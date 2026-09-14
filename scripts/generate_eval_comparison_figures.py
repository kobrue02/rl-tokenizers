"""Generates thesis-ready TikZ/pgfplots grouped-bar figures comparing
tokenizers on DOWNSTREAM pretraining-eval results -- as opposed to
scripts.generate_tikz_figures, which plots tokenizer-INTRINSIC metrics
(fertility/compression/parity from systems/*/evaluate.py) and never touches
XNLI/XCOPA/FLORES-MT/NER/POS/etc. at all.

Two independent inputs, each optional (pass either or both):
  --decoder-input  scripts.combine_decoder_results's combined JSON
                    ({label: {benchmark: result}}, benchmark in
                    xnli/xcopa/blimp/cola/squad/flores_mt).
  --encoder-input  scripts.combine_encoder_results's combined JSON
                    ({label: {benchmark_or_task: record}}, record["result"]
                    covering pppl/retrieval/roundtrip/ner/pos/taxi1500/sib200).

Each side gets TWO figures, not one -- cramming every benchmark onto a
single y-axis would be misleading, since they don't share a scale or a
"higher is better" direction:
  decoder:  decoder_classification/ (xnli/xcopa/blimp accuracy, squad F1,
            cola MCC -- all roughly [-1,1]-ish, higher is better)
            decoder_flores_mt/      (BLEU + chrF, ~[0,100] scale)
  encoder:  encoder_classification/ (retrieval/roundtrip/ner/pos/taxi1500/
            sib200 -- all [0,1], higher is better)
            encoder_pseudoperplexity/ (unbounded positive, LOWER is better --
            kept off the classification axis so it can't dwarf/be dwarfed by
            a [0,1] bar depending on the checkpoint's actual perplexity)

Same house style as generate_tikz_figures.py: no matplotlib, pure Python
writing pgfplots .dat tables + a standalone/_body .tex pair per figure (see
that module's own _write_standalone_and_body). One categorical color per
LABEL (not per tokenizer family -- these figures compare a handful of runs
by name, e.g. "bpe" vs "fanta", not dozens of external tokenizers), assigned
deterministically by sorted label order so re-running with the same labels
always gets the same colors.

Usage:
    python3 -m scripts.generate_eval_comparison_figures \\
        --decoder-input results/decoder_comparison.json \\
        --encoder-input results/encoder_comparison.json \\
        --output-dir figures/tikz
    python3 -m scripts.generate_eval_comparison_figures -c configs/some_config.yml

No local LaTeX install to compile-test against -- verify with a real
compiler (Overleaf is fine) before trusting the output.
"""

import json
import os

from common.config_file import parse_args_with_config
from scripts.generate_encoder_comparison_table import _headline_value

# Colorblind-safe categorical palette (Okabe-Ito), assigned to labels by
# sorted order -- deterministic across runs, cycles if there are ever more
# labels than colors (unlikely for a handful of tokenizer comparisons).
_LABEL_PALETTE = [
    (0, 114, 178),   # blue
    (230, 159, 0),   # orange
    (0, 158, 115),   # green
    (213, 94, 0),    # vermillion
    (204, 121, 167), # purple
    (86, 180, 233),  # sky blue
    (240, 228, 66),  # yellow
]

# (benchmark key, metric spec, display name) -- spec is an exact key for
# decoder benchmarks (xnli/xcopa/blimp/squad/cola always return one fixed
# shape) or a key SUFFIX for encoder finetune benchmarks (ner/pos/taxi1500/
# sib200's --eval-langs=all sweeps produce one eval_{lang}_<metric> key per
# language -- see generate_encoder_comparison_table._headline_value, reused
# here unchanged so both places compute "the mean across every language a
# label was evaluated on" identically).
_DECODER_CLASSIFICATION_SPECS = [
    ("xnli", "accuracy", "XNLI"),
    ("xcopa", "accuracy", "XCOPA"),
    ("blimp", "accuracy", "BLiMP"),
    ("squad", "f1", "SQuAD (F1)"),
    ("cola", "mcc", "CoLA (MCC)"),
]
_ENCODER_CLASSIFICATION_SPECS = [
    ("retrieval", "top10_accuracy", "Retrieval"),
    ("roundtrip", "roundtrip_accuracy", "Roundtrip"),
    ("ner", "_f1", "NER (F1)"),
    ("pos", "_accuracy", "POS"),
    ("taxi1500", "_macro_f1", "Taxi1500"),
    ("sib200", "_macro_f1", "SIB-200"),
]


def esc(s):
    """LaTeX-safe for use as a literal label."""
    return s.replace("_", r"\_")


def label_key(label):
    """Filesystem/macro-safe stand-in for a label (used in .dat filenames
    and TikZ color macro names)."""
    return label.replace("/", "_").replace(" ", "_").replace(".", "_")


def _write_standalone_and_body(name, preamble_lines, body_lines, out_dir):
    """Mirrors generate_tikz_figures._write_standalone_and_body exactly
    (not imported from there to avoid a cross-module coupling on a private
    helper) -- see that function's own docstring for why both a standalone
    and a _body variant are written."""
    full_tex = "\n".join(preamble_lines + [r"\begin{document}"] + body_lines + [r"\end{document}"])
    body_tex = "\n".join(body_lines)
    with open(os.path.join(out_dir, f"fig_{name}.tex"), "w", encoding="utf-8") as f:
        f.write(full_tex)
    with open(os.path.join(out_dir, f"fig_{name}_body.tex"), "w", encoding="utf-8") as f:
        f.write(body_tex)
    return full_tex, body_tex


def gen_grouped_bar_tex(
    data, categories, out_dir, fig_name, ylabel, data_prefix="", note=None, bar_width_total_pt=18,
):
    """data: {label: {category_key: value}} (a label/category combination
    with no value is simply omitted, not fabricated as 0 -- see the
    present_labels filtering below). categories: ordered [(key, display), ...].

    One clustered ybar chart, categories on the x-axis, one bar series per
    label. `note` (e.g. "lower is better") is appended to ylabel in
    parentheses if given.

    Labels with NO data for any category in this figure are dropped from
    present_labels entirely, not just given empty bars: pgfplots silently
    drops a completely-empty-table addplot from its own legend numbering,
    which shifts every LATER \\addlegendentry to mislabel the next real
    plot (see generate_tikz_figures.gen_indigenous_panel_parity_bars_tex's
    own docstring for the same fix, applied here identically)."""
    labels = sorted(data)
    present_labels = [l for l in labels if any(data[l].get(k) is not None for k, _ in categories)]
    dropped_labels = [l for l in labels if l not in present_labels]
    if dropped_labels:
        print(f"  {fig_name}: label(s) with NO data for this figure, omitted: {dropped_labels}")
    if not present_labels:
        print(f"  {fig_name}: no label has any data for this figure -- skipping")
        return None

    # Categories no PRESENT label has any data for (e.g. blimp/squad/cola
    # when only xnli/xcopa/flores_mt were ever run) are dropped too, not
    # rendered as an empty x-axis slot with no bars -- `categories` is the
    # full set this figure COULD show, not a claim every run covers all of it.
    present_categories = [(k, d) for k, d in categories if any(data[l].get(k) is not None for l in present_labels)]
    dropped_categories = [k for k, _ in categories if k not in {k2 for k2, _ in present_categories}]
    if dropped_categories:
        print(f"  {fig_name}: category/categories with NO data from any present label, omitted: {dropped_categories}")
    categories = present_categories

    colors = {l: _LABEL_PALETTE[i % len(_LABEL_PALETTE)] for i, l in enumerate(present_labels)}
    color_macro = {l: f"evalCol{label_key(l)}" for l in present_labels}

    for l in present_labels:
        path = os.path.join(out_dir, f"bar_{fig_name}_{label_key(l)}.dat")
        with open(path, "w", encoding="utf-8") as f:
            f.write("category value\n")
            for key, _ in categories:
                v = data[l].get(key)
                if v is None:
                    continue
                f.write(f"{key} {v:.4f}\n")

    preamble = [r"\documentclass{standalone}", r"\usepackage{pgfplots}", r"\pgfplotsset{compat=1.18}"]
    for l in present_labels:
        r_, g_, b_ = colors[l]
        preamble.append(r"\definecolor{%s}{RGB}{%d,%d,%d}" % (color_macro[l], r_, g_, b_))

    symbolic_coords = ",".join(k for k, _ in categories)
    xticklabels = ", ".join(f"{{{esc(d)}}}" for _, d in categories)
    full_ylabel = f"{ylabel} ({note})" if note else ylabel

    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    ybar, width=%.1fcm, height=6.5cm," % max(8.0, len(categories) * 2.2),
        r"    bar width=%dpt," % max(3, bar_width_total_pt // max(len(present_labels), 1)),
        r"    symbolic x coords={%s}," % symbolic_coords,
        r"    xtick=data,",
        r"    xticklabels={%s}," % xticklabels,
        r"    x tick label style={font=\small},",
        r"    ylabel={%s}," % esc(full_ylabel),
        r"    ymin=0,",
        r"    enlarge x limits=%.3f," % (0.6 / max(len(categories), 1)),
        r"    legend style={at={(1.02,1)}, anchor=north west, font=\scriptsize, draw=none, fill=none},",
        r"    axis y line*=left, axis x line*=bottom,",
        r"]",
    ]
    for l in present_labels:
        col = color_macro[l]
        body.append(
            r"\addplot+[ybar, fill=%s, draw=%s] table [x=category, y=value] {%sbar_%s_%s.dat};"
            % (col, col, data_prefix, fig_name, label_key(l))
        )
        body.append(r"\addlegendentry{%s}" % esc(l))
    body += [r"\end{axis}", r"\end{tikzpicture}"]

    full_tex, _ = _write_standalone_and_body(fig_name, preamble, body, out_dir)
    return full_tex


def load_decoder_combined(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_encoder_combined(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_decoder_figures(input_path, base_out_dir):
    combined = load_decoder_combined(input_path)

    classification_data = {}
    for label, records in combined.items():
        row = {}
        for benchmark, metric, _ in _DECODER_CLASSIFICATION_SPECS:
            result = records.get(benchmark)
            if result is not None and metric in result:
                row[benchmark] = result[metric]
        if row:
            classification_data[label] = row
    classification_categories = [(b, d) for b, _, d in _DECODER_CLASSIFICATION_SPECS]
    out_dir = os.path.join(base_out_dir, "decoder_classification")
    os.makedirs(out_dir, exist_ok=True)
    gen_grouped_bar_tex(
        classification_data, classification_categories, out_dir, "decoder_classification",
        ylabel="Score (accuracy / F1 / MCC)",
    )

    flores_data = {}
    for label, records in combined.items():
        result = records.get("flores_mt")
        if result is None:
            continue
        row = {}
        if "bleu" in result:
            row["bleu"] = result["bleu"]
        if "chrf" in result:
            row["chrf"] = result["chrf"]
        if row:
            flores_data[label] = row
    out_dir = os.path.join(base_out_dir, "decoder_flores_mt")
    os.makedirs(out_dir, exist_ok=True)
    gen_grouped_bar_tex(
        flores_data, [("bleu", "BLEU"), ("chrf", "chrF")], out_dir, "decoder_flores_mt",
        ylabel="FLORES-MT score",
    )

    print(f"decoder: {len(combined)} label(s) -- wrote decoder_classification/ and decoder_flores_mt/")


def generate_encoder_figures(input_path, base_out_dir):
    combined = load_encoder_combined(input_path)

    classification_data = {}
    for label, records in combined.items():
        row = {}
        for benchmark, spec, _ in _ENCODER_CLASSIFICATION_SPECS:
            record = records.get(benchmark)
            if record is None:
                continue
            value = _headline_value(record["result"], spec)
            if value is not None:
                row[benchmark] = value
        if row:
            classification_data[label] = row
    classification_categories = [(b, d) for b, _, d in _ENCODER_CLASSIFICATION_SPECS]
    out_dir = os.path.join(base_out_dir, "encoder_classification")
    os.makedirs(out_dir, exist_ok=True)
    gen_grouped_bar_tex(
        classification_data, classification_categories, out_dir, "encoder_classification",
        ylabel="Score (accuracy / F1)",
    )

    pppl_data = {}
    for label, records in combined.items():
        record = records.get("pppl")
        if record is None:
            continue
        value = record["result"].get("pseudoperplexity")
        if value is not None:
            pppl_data[label] = {"pppl": value}
    out_dir = os.path.join(base_out_dir, "encoder_pseudoperplexity")
    os.makedirs(out_dir, exist_ok=True)
    gen_grouped_bar_tex(
        pppl_data, [("pppl", "Pseudoperplexity")], out_dir, "encoder_pseudoperplexity",
        ylabel="Pseudoperplexity", note="lower is better",
    )

    print(f"encoder: {len(combined)} label(s) -- wrote encoder_classification/ and encoder_pseudoperplexity/")


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate grouped-bar TikZ/pgfplots figures from combined decoder/encoder eval results."
    )
    parser.add_argument(
        "--decoder-input", type=str, default=None,
        help="scripts.combine_decoder_results combined JSON (xnli/xcopa/blimp/cola/squad/flores_mt)",
    )
    parser.add_argument(
        "--encoder-input", type=str, default=None,
        help="scripts.combine_encoder_results combined JSON (pppl/retrieval/roundtrip/ner/pos/taxi1500/sib200)",
    )
    parser.add_argument("--output-dir", type=str, default="figures/tikz", help="base output directory")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    if not args.decoder_input and not args.encoder_input:
        raise ValueError("pass at least one of --decoder-input / --encoder-input")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.decoder_input:
        generate_decoder_figures(args.decoder_input, args.output_dir)
    if args.encoder_input:
        generate_encoder_figures(args.encoder_input, args.output_dir)


if __name__ == "__main__":
    main()
