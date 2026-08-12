"""CLI: apply a trained (frozen) policy checkpoint to a real corpus to build
the final tokenizer vocabulary. Separate entry point from fairtok.cli (which
trains the policy on the Phase 1 fairness data) -- this step needs no
parallel groups, no reward, no gradients, just the corpus you'll actually
pretrain the downstream LM on.
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from .inference import build_and_save_vocab, load_checkpoint


def _iter_corpus(path):
    """Generic line-delimited text loader: one document per line, from a single
    file or every .txt file in a directory. A stand-in for whatever the real
    pretraining-corpus loader will be -- this project only has the Phase 1
    fairness-training data wired up (see common.oldi_data), not the actual
    downstream pretraining corpus, which lives outside this repo."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.suffix == ".txt")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Apply a trained policy checkpoint to a corpus to build the final tokenizer vocabulary."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="policy checkpoint saved by fairtok.train (GRPOConfig.output_dir)",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="a text file, or a directory of .txt files, one document per line",
    )
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="threshold boundary probability at 0.5 (default) instead of sampling -- reproducible given the same corpus",
    )
    parser.add_argument("--vocab-out", type=str, default="corpus_vocab.json")
    parser.add_argument(
        "--vocab-stats-out", type=str, default="corpus_vocab_stats.json"
    )
    parser.add_argument("--vocab-preview", type=int, default=20)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    policy = load_checkpoint(args.checkpoint)
    texts = _iter_corpus(args.corpus)

    token_freq, entries = build_and_save_vocab(
        policy,
        texts,
        args.vocab_size,
        args.vocab_out,
        args.vocab_stats_out,
        deterministic=args.deterministic,
        progress=lambda it, desc: tqdm(it, desc=desc, unit="doc"),
    )

    distinct_spans = sum(len(c) for c in token_freq.values())
    print(
        f"\nvocab entries kept: {len(entries)} (of {distinct_spans} distinct spans seen)"
    )

    if args.vocab_preview:
        print(
            f"\ntop {min(args.vocab_preview, len(entries))} vocab entries by frequency:"
        )
        for span, total, _ in entries[: args.vocab_preview]:
            print(f"  {total:6d}  {span!r}")

    if args.vocab_out:
        print(f"\nsaved vocab ({len(entries)} entries) to {args.vocab_out}")
    if args.vocab_stats_out:
        print(f"saved per-entry frequency/source stats to {args.vocab_stats_out}")

    return token_freq, entries


if __name__ == "__main__":
    main()
