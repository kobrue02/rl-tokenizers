"""Merges multiple systems.pretraining.cli_eval --output JSON files into one
combined comparison, keyed by "label" then by benchmark name.

Unlike scripts.combine_encoder_results (each encoder eval CLI invocation
covers exactly one benchmark/task), cli_eval.py can score several benchmarks
in a single job (--benchmark xnli,xcopa,flores_mt -> one combined "results"
dict, see cli_eval.py's own docstring) -- this flattens that per-file
{benchmark: result} dict into the same {label: {benchmark: result}} shape
combine_encoder_results produces, so a real multi-tokenizer comparison can
merge files that each cover a different, possibly overlapping, subset of
benchmarks for a given label without one overwriting another's benchmark.

Usage:
    python3 -m scripts.combine_decoder_results \\
        --input results/all_bpe_large.json results/all_fanta_large.json \\
        --output results/decoder_comparison.json

Feed the result straight into scripts.generate_eval_comparison_figures.
"""

import argparse
import json

from common.config_file import parse_args_with_config


def combine_decoder_results(paths):
    combined = {}
    sources_by_key = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        label = record["label"]
        results = record.get("results")
        if results is None:
            raise ValueError(
                f"{path}: record has no 'results' key -- not a recognized "
                "systems.pretraining.cli_eval output file (re-run eval after "
                "the --label/--results wrapper was added, see cli_eval.py's docstring)"
            )
        for benchmark, result in results.items():
            full_key = (label, benchmark)
            if full_key in sources_by_key:
                print(
                    f"warning: (label={label!r}, benchmark={benchmark!r}) appears in more than one "
                    f"input file ({sources_by_key[full_key]!r} and {path!r}) -- keeping the later one ({path!r})"
                )
            combined.setdefault(label, {})[benchmark] = result
            sources_by_key[full_key] = path
    return combined


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge multiple systems.pretraining.cli_eval --output JSON files "
        "into one combined comparison, keyed by label then benchmark."
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="one or more result JSON file paths to merge (e.g. results/all_bpe_large.json "
        "results/all_fanta_large.json)",
    )
    parser.add_argument("--output", type=str, required=True, help="where to write the combined JSON")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    combined = combine_decoder_results(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    n_records = sum(len(v) for v in combined.values())
    print(f"wrote {len(combined)} label(s) / {n_records} benchmark record(s) to {args.output}")


if __name__ == "__main__":
    main()
