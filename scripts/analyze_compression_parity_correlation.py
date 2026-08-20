"""Answers the Token Tax chapter's \\S{sec:tt-landscape} TODO (fig:fairness-
landscape), described there as "the load-bearing result for the thesis":
are compression and (anchor-invariant) parity spread correlated across
tokenizers, and which tokenizers are existence proofs that the two axes
come apart -- efficient (high avg_compression) but unequal (high spread),
or the reverse (low compression, low spread)?

Both Pearson (linear) and Spearman (rank) correlation are reported: if the
two axes were tightly correlated, a fairness objective would be
unnecessary (optimising compression would deliver parity for free) --
Pearson is the more natural statistic for that specific claim, Spearman is
reported alongside as a check that the conclusion doesn't depend on
assuming a linear relationship.

Outlier tokenizers are found by splitting at each axis's OWN median (not a
regression residual): the two quadrants that would be UNDER-populated if
compression and spread were tightly coupled -- (high compression, high
spread) and (low compression, low spread) -- are exactly the quadrants
that constitute an "axes come apart" existence proof; the other two
quadrants (high compression+low spread, low compression+high spread) are
what a tight positive correlation would predict, so they aren't evidence
of independence on their own.

Usage:
    python3 -m scripts.analyze_compression_parity_correlation \\
        --input results/all_tokenizers_comparison.json
"""

import argparse
import statistics

from scipy import stats

from scripts.generate_tikz_figures import load_rows


def _quadrant_outliers(rows, top_n=5):
    compressions = [r["avg_compression"] for r in rows]
    spreads = [r["spread"] for r in rows]
    med_compression = statistics.median(compressions)
    med_spread = statistics.median(spreads)

    efficient_but_unequal = [
        r for r in rows if r["avg_compression"] > med_compression and r["spread"] > med_spread
    ]
    equal_but_inefficient = [
        r for r in rows if r["avg_compression"] <= med_compression and r["spread"] <= med_spread
    ]
    efficient_but_unequal.sort(key=lambda r: (-r["avg_compression"], -r["spread"]))
    equal_but_inefficient.sort(key=lambda r: (r["avg_compression"], r["spread"]))
    return med_compression, med_spread, efficient_but_unequal[:top_n], equal_but_inefficient[:top_n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    parser.add_argument("--top-n", type=int, default=5, help="outlier tokenizers to list per quadrant")
    args = parser.parse_args()

    rows, _models = load_rows(args.input)
    compressions = [r["avg_compression"] for r in rows]
    spreads = [r["spread"] for r in rows]

    pearson_r, pearson_p = stats.pearsonr(compressions, spreads)
    spearman_rho, spearman_p = stats.spearmanr(compressions, spreads)

    print(f"n={len(rows)}")
    print(f"Pearson r={pearson_r:+.3f} (p={pearson_p:.3f})")
    print(f"Spearman rho={spearman_rho:+.3f} (p={spearman_p:.3f})")
    r2 = pearson_r ** 2
    print(f"r^2={r2:.3f} -> compression explains {r2*100:.1f}% of the variance in spread")

    med_compression, med_spread, efficient_but_unequal, equal_but_inefficient = _quadrant_outliers(
        rows, top_n=args.top_n
    )
    print(f"\nmedian avg_compression={med_compression:.3f}, median spread={med_spread:.3f}")

    print(f"\nefficient but unequal (compression > median AND spread > median), top {args.top_n}:")
    for r in efficient_but_unequal:
        print(f"  {r['name']:45s} compression={r['avg_compression']:.3f} spread={r['spread']:.3f}")

    print(f"\nequal but inefficient (compression <= median AND spread <= median), top {args.top_n}:")
    for r in equal_but_inefficient:
        print(f"  {r['name']:45s} compression={r['avg_compression']:.3f} spread={r['spread']:.3f}")


if __name__ == "__main__":
    main()
