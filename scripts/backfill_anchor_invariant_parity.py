"""Backfills token_parity_gm/token_parity_spread (anchor-invariant -- a fixed
anchor's ranking can flip depending on which language you pick) into an existing
systems/*/evaluate.py --output JSON file, computed purely from the file's own
already-stored token_parity (no re-tokenization or network calls). Needed only
for results computed before this metric existed. Modifies --input files IN
PLACE; entries that already have both fields are left untouched.

Usage:
    python3 -m scripts.backfill_anchor_invariant_parity --input results/hf_frontier_comparison.json
"""

import argparse
import json

from common.config_file import parse_args_with_config
from common.eval.parity import anchor_invariant_parity


def backfill(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    updated = []
    skipped = []
    for key, entry in data.items():
        if key == "_failed" or not isinstance(entry, dict):
            continue
        if "token_parity_gm" in entry and "token_parity_spread" in entry:
            skipped.append(key)
            continue
        if "token_parity" not in entry:
            continue  # nothing to derive it from
        gm, spread = anchor_invariant_parity(entry["token_parity"])
        entry["token_parity_gm"] = gm
        entry["token_parity_spread"] = spread
        updated.append(key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"{path}: backfilled {len(updated)} entr{'y' if len(updated) == 1 else 'ies'}: {updated}")
    if skipped:
        print(f"{path}: already had it, left unchanged: {skipped}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Backfill token_parity_gm/token_parity_spread into existing evaluate.py --output JSON files."
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="one or more results JSON files to backfill IN PLACE",
    )
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    for path in args.input:
        backfill(path)


if __name__ == "__main__":
    main()
