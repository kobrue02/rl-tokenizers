"""Merges multiple systems.pretraining.encoder_cli_eval --output /
encoder_cli_finetune --results-output JSON files into one combined
comparison, keyed by "label" then by "benchmark"/"task".

Unlike scripts/combine_eval_results.py (a flat one-record-per-tokenizer
merge for systems/*/evaluate.py's own output shape, where a single file
already covers everything for one tokenizer), each encoder eval/finetune
CLI invocation only ever covers ONE benchmark/task at a time (pppl,
retrieval, roundtrip, ner, pos, taxi1500, sib200) -- see
encoder_cli_eval.py/encoder_cli_finetune.py's own --benchmark/--task
dispatch. A real multi-tokenizer comparison therefore comes from MANY
separate result files per tokenizer (one per benchmark/task run) that need
merging under the SAME label without one overwriting another's
benchmark/task -- this does that nested merge, warning (not silently
overwriting) on a genuine (label, benchmark/task) collision.

Usage:
    python3 -m scripts.combine_encoder_results \\
        --input results/encoder/pppl_bpe.json results/encoder/retrieval_bpe.json results/encoder/ner_bpe.json \\
                results/encoder/pppl_fanta.json results/encoder/ner_fanta.json \\
        --output results/encoder_comparison.json

Feed the result straight into scripts.generate_encoder_comparison_table.
"""

import argparse
import json

from common.config_file import parse_args_with_config


def combine_encoder_results(paths):
    combined = {}
    sources_by_key = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        label = record["label"]
        key = record.get("benchmark") or record.get("task")
        if key is None:
            raise ValueError(
                f"{path}: record has neither 'benchmark' nor 'task' -- not a recognized "
                "encoder_cli_eval/encoder_cli_finetune output file"
            )
        full_key = (label, key)
        if full_key in sources_by_key:
            print(
                f"warning: (label={label!r}, benchmark/task={key!r}) appears in more than one "
                f"input file ({sources_by_key[full_key]!r} and {path!r}) -- keeping the later one ({path!r})"
            )
        combined.setdefault(label, {})[key] = record
        sources_by_key[full_key] = path
    return combined


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge multiple encoder_cli_eval/encoder_cli_finetune --output JSON files "
        "into one combined comparison, keyed by label then benchmark/task."
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="one or more result JSON file paths to merge (e.g. results/encoder/pppl_bpe.json "
        "results/encoder/ner_bpe.json results/encoder/pppl_fanta.json)",
    )
    parser.add_argument("--output", type=str, required=True, help="where to write the combined JSON")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    combined = combine_encoder_results(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    n_records = sum(len(v) for v in combined.values())
    print(f"wrote {len(combined)} label(s) / {n_records} benchmark-task record(s) to {args.output}")


if __name__ == "__main__":
    main()
