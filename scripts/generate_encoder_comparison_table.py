"""Renders scripts.combine_encoder_results's own combined JSON as markdown
comparison tables -- one row per label (tokenizer), covering whichever
benchmarks/tasks (pppl, retrieval, roundtrip, ner, pos, taxi1500, sib200)
each label actually has results for.

Two tables, since a --eval-langs=all sweep (encoder_cli_finetune.py) can
produce dozens to hundreds of per-language metric keys (eval_deu_f1,
eval_fra_f1, ...) that would make one table unreadable:

  SUMMARY: one column per benchmark/task, one headline number per cell --
  for ner/pos/taxi1500/sib200 that's the MEAN across every language a
  label was evaluated on (one value if only a single --eval-lang was used,
  a real cross-language average if --eval-langs=all was) -- see
  _headline_value's own docstring. Missing cells (a label that hasn't run
  that benchmark/task yet) render as "--", never 0 or blank -- silently
  implying "not evaluated" would read the same as "scored zero".

  DETAILED: every raw metric key that appears in ANY label's results,
  flattened as "{benchmark_or_task}.{metric_key}" columns -- the full
  per-language breakdown for anyone who wants to dig past the summary mean.

Usage:
    python3 -m scripts.generate_encoder_comparison_table --input results/encoder_comparison.json \\
        --output results/encoder_comparison.md
"""

import argparse
import json

from common.config_file import parse_args_with_config

# exact key (pppl/retrieval/roundtrip's results are always this one fixed
# key) OR a key SUFFIX (ner/pos/taxi1500/sib200's finetune results, which
# can be a single eval_f1-style key or several eval_{lang}_f1-style keys
# from a --eval-langs=all sweep -- see _headline_value).
_HEADLINE_METRIC = {
    "pppl": "pseudoperplexity",
    "retrieval": "top10_accuracy",
    "roundtrip": "roundtrip_accuracy",
    "ner": "_f1",
    "pos": "_accuracy",
    "taxi1500": "_macro_f1",
    "sib200": "_macro_f1",
}


def _headline_value(result, spec):
    """spec is an exact key when there's only ever one fixed shape (pppl/
    retrieval/roundtrip); ner/pos/taxi1500/sib200 use a suffix instead,
    since encoder_cli_finetune.py's --eval-langs/--eval-configs/
    --eval-lang-scripts=all sweeps produce one eval_{lang}_<metric> key per
    language rather than a single eval_<metric> -- averaging over every
    matching key handles both the single- and multi-language shapes with
    the same code, and IS the natural "how did this tokenizer do across
    every language it was evaluated on" summary."""
    if spec in result:
        return result[spec]
    matches = [v for k, v in result.items() if isinstance(v, (int, float)) and k.endswith(spec)]
    return sum(matches) / len(matches) if matches else None


def _fmt(value):
    return "--" if value is None else f"{value:.4f}"


def build_summary_table(combined):
    benchmarks = sorted({key for records in combined.values() for key in records})
    header = ["label"] + benchmarks
    rows = []
    for label in sorted(combined):
        row = [label]
        for benchmark in benchmarks:
            record = combined[label].get(benchmark)
            if record is None:
                row.append("--")
                continue
            spec = _HEADLINE_METRIC.get(benchmark)
            value = _headline_value(record["result"], spec) if spec else None
            row.append(_fmt(value))
        rows.append(row)
    return header, rows


def build_detailed_table(combined):
    columns = sorted(
        {
            f"{benchmark}.{metric}"
            for records in combined.values()
            for benchmark, record in records.items()
            for metric in record["result"]
        }
    )
    header = ["label"] + columns
    rows = []
    for label in sorted(combined):
        row = [label]
        for column in columns:
            benchmark, metric = column.split(".", 1)
            record = combined[label].get(benchmark)
            value = record["result"].get(metric) if record else None
            row.append(_fmt(value) if isinstance(value, (int, float)) else ("--" if value is None else str(value)))
        rows.append(row)
    return header, rows


def render_markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_report(combined):
    summary_header, summary_rows = build_summary_table(combined)
    detailed_header, detailed_rows = build_detailed_table(combined)
    return (
        "## Summary\n\n"
        + render_markdown_table(summary_header, summary_rows)
        + "\n\n## Detailed\n\n"
        + render_markdown_table(detailed_header, detailed_rows)
        + "\n"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Render scripts.combine_encoder_results's combined JSON as markdown comparison tables."
    )
    parser.add_argument("--input", type=str, required=True, help="combined JSON from scripts.combine_encoder_results")
    parser.add_argument("--output", type=str, required=True, help="where to write the markdown report")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    with open(args.input, encoding="utf-8") as f:
        combined = json.load(f)
    report = render_report(combined)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote comparison tables for {len(combined)} label(s) to {args.output}")


if __name__ == "__main__":
    main()
