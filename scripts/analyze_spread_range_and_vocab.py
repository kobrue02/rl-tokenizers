"""Answers two open TODOs in the Token Tax chapter's \\S{sec:tt-spread}
("How Far Apart Are Tokenizers?"):

  1. The pooled fig:spread-leaderboard TODO: which tokenizer has the
     tightest/widest anchor-invariant spread, and what is the factor
     between them?
  2. The fig:spread-leaderboard-frontier TODO: does the spread ranking
     track vocabulary size, restricted to the frontier subset where vocab
     size is actually documented/comparable?

Both questions are also answered for the FULL pooled set (not just the
frontier subset) for completeness, using scripts/backfill_vocab_sizes.py's
output where available -- correlation there is noisier since it mixes
byte-level BPE, SentencePiece, WordPiece and this project's own span-family
tokenizers, which is exactly why the frontier-only cut is the more honest
place to check the vocab-size claim (see the chapter TODO's own reasoning).

Spearman (not Pearson) is the primary statistic: the TODO asks whether the
RANKING tracks vocabulary size, not whether the relationship is linear.

Usage:
    python3 -m scripts.analyze_spread_range_and_vocab \\
        --input results/all_tokenizers_comparison.json \\
        --vocab-sizes results/vocab_sizes.json \\
        --hf-frontier-keys results/hf_frontier_comparison.json
"""

import argparse
import json

from scipy import stats

from scripts.generate_tikz_figures import load_rows

# Same "frontier" definition as scripts/generate_scoped_leaderboards.py's own
# generate_hf_frontier_leaderboard: hf_frontier_comparison.json's key set
# (every HF-Hub-hosted tokenizer actually evaluated) plus claude-opus-5
# explicitly (not HF-Hub-hosted, but unambiguously a frontier model).
_EXTRA_FRONTIER_KEYS = {"claude-opus-5"}


def _frontier_keys(hf_frontier_keys_path):
    with open(hf_frontier_keys_path, encoding="utf-8") as f:
        return set(json.load(f).keys()) | set(_EXTRA_FRONTIER_KEYS)


def _range_and_factor(rows):
    """rows: pre-sorted ascending by spread (load_rows's own convention)."""
    tightest, widest = rows[0], rows[-1]
    factor = widest["spread"] / tightest["spread"] if tightest["spread"] else float("inf")
    return tightest, widest, factor


def _spread_vocab_correlation(rows, vocab_sizes):
    """rows: any subset of load_rows's output. Returns (n, rho, pvalue) over
    entries with a known (non-null) vocab_size; (0, None, None) if fewer
    than 3 such entries (scipy.stats.spearmanr needs at least 2, and a
    2-point correlation is meaningless -- floored at 3 here)."""
    paired = [
        (r["spread"], vocab_sizes[r["name"]]["vocab_size"])
        for r in rows
        if r["name"] in vocab_sizes and vocab_sizes[r["name"]]["vocab_size"] is not None
    ]
    if len(paired) < 3:
        return len(paired), None, None
    spreads, vocabs = zip(*paired)
    rho, pvalue = stats.spearmanr(spreads, vocabs)
    return len(paired), rho, pvalue


def _report(label, rows, vocab_sizes):
    print(f"\n=== {label} (n={len(rows)}) ===")
    tightest, widest, factor = _range_and_factor(rows)
    print(f"tightest spread: {tightest['name']} (spread={tightest['spread']:.3f})")
    print(f"widest spread:   {widest['name']} (spread={widest['spread']:.3f})")
    print(f"factor (widest/tightest): {factor:.2f}x")

    n, rho, pvalue = _spread_vocab_correlation(rows, vocab_sizes)
    if rho is None:
        print(f"spread-vs-vocab_size correlation: skipped (only {n} entries with known vocab_size)")
    else:
        print(f"spread-vs-vocab_size Spearman rho={rho:+.3f} (p={pvalue:.3f}, n={n})")
        strength = "no" if abs(rho) < 0.2 else "weak" if abs(rho) < 0.4 else "moderate" if abs(rho) < 0.6 else "strong"
        direction = "larger vocab -> wider spread" if rho > 0 else "larger vocab -> tighter spread"
        print(f"  -> {strength} correlation ({direction} if significant); "
              f"{'significant' if pvalue < 0.05 else 'NOT significant'} at alpha=0.05")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    parser.add_argument("--vocab-sizes", default="results/vocab_sizes.json")
    parser.add_argument("--hf-frontier-keys", default="results/hf_frontier_comparison.json")
    args = parser.parse_args()

    rows, _models = load_rows(args.input)
    with open(args.vocab_sizes, encoding="utf-8") as f:
        vocab_sizes = json.load(f)

    _report("pooled (all tokenizers)", rows, vocab_sizes)

    frontier_keys = _frontier_keys(args.hf_frontier_keys)
    frontier_rows = [r for r in rows if r["name"] in frontier_keys]
    frontier_rows.sort(key=lambda r: r["spread"])
    _report("frontier subset", frontier_rows, vocab_sizes)


if __name__ == "__main__":
    main()
