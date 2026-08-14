"""Merges multiple systems/*/evaluate.py --output JSON result files into one
combined comparison -- e.g. systems/hf_frontier/evaluate.py's own
results/hf_frontier_comparison.json (renyi/gini/compression/fertility/
token_parity per repo) and systems/claude_tokenizer/evaluate.py's own
results/claude_comparison.json (compression/fertility/token_parity only --
see that module's own docstring for why renyi/gini aren't available for it).

This works as a plain dict merge, keyed by tokenizer/model name, because
every source already writes this SAME per-tokenizer results shape (see
common.eval.cross_tokenizer.evaluate_on_groups's own docstring) -- nothing
here reformats or recomputes anything, it just combines files that already
agree on shape. Warns (doesn't silently drop) on a name collision between
input files, and reports which combined entries lack renyi/gini so a
downstream consumer (e.g. a dashboard) doesn't mistake a genuinely
unavailable metric for a zero/perfect one.

Usage:
    python3 combine_eval_results.py \\
        --input results/hf_frontier_comparison.json results/claude_comparison.json \\
        --output results/all_tokenizers_comparison.json
"""

import argparse
import json

from common.config_file import parse_args_with_config


def combine_results(paths):
    combined = {}
    failed = {}
    sources_by_key = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if key == "_failed":
                failed.update(value)
                continue
            if key in combined:
                print(
                    f"warning: {key!r} appears in more than one input file "
                    f"({sources_by_key[key]!r} and {path!r}) -- keeping the later one ({path!r})"
                )
            combined[key] = value
            sources_by_key[key] = path

    missing_renyi = [k for k, v in combined.items() if not v.get("renyi")]
    if missing_renyi:
        plural = "y" if len(missing_renyi) == 1 else "ies"
        print(
            f"note: {len(missing_renyi)} entr{plural} have no renyi/gini (genuinely unavailable "
            f"from their own source, not zero -- see that evaluate.py's own docstring): {missing_renyi}"
        )

    if failed:
        combined["_failed"] = failed
    return combined


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge multiple systems/*/evaluate.py --output JSON files into one combined comparison."
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="one or more result JSON file paths to merge (e.g. results/hf_frontier_comparison.json "
        "results/claude_comparison.json)",
    )
    parser.add_argument("--output", type=str, required=True, help="where to write the combined JSON")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    combined = combine_results(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    n = len(combined) - (1 if "_failed" in combined else 0)
    print(f"wrote {n} tokenizer entries to {args.output}")


if __name__ == "__main__":
    main()
