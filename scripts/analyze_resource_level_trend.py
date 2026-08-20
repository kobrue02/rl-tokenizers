"""Answers the Token Tax chapter's \\S{sec:tt-resource} TODO
(fig:resource-level-trend), the "systematicity" result: does token parity
(English-relative) track Joshi et al. (2020)'s linguistic resource level,
and how strong/consistent is that trend -- not just for the pooled
average, but across individual tokenizers (fig:resource-level-trend itself
draws one line per tokenizer; this reports whether most of those lines
individually point the same direction, not just the pooled mean).

Four numbers, in increasing order of how much they commit to "systematic":
  1. Pooled mean token_parity per level (what the figure's y-axis position
     comes from, averaged over every (tokenizer, language) pair at that level).
  2. Pooled Spearman rho (level vs. every individual (tokenizer, language)
     parity value) -- the overall trend strength.
  3. A Kruskal-Wallis test across the 6 levels on those same pooled values --
     tests whether the 6 level-groups' distributions differ at all (not just
     monotonically), a complementary significance check to the correlation.
  4. Per-tokenizer Spearman rho (level vs. that tokenizer's OWN per-language
     parity), summarised as how many tokenizers individually show a
     significant (p<0.05) negative trend -- this is what makes the result
     "structural" (\\S{sec:tt-implications}) rather than a pooling artifact
     of a few extreme tokenizers.

Usage:
    python3 -m scripts.analyze_resource_level_trend \\
        --input results/all_tokenizers_comparison.json
"""

import argparse
import statistics
from collections import defaultdict

from scipy import stats

from common.data.lang2tax import load_resource_levels
from scripts.generate_tikz_figures import load_rows

_RESOURCE_LEVEL_LABELS = {
    0: "0: Left-Behinds", 1: "1: Scraping-Bys", 2: "2: Hopefuls",
    3: "3: Rising Stars", 4: "4: Underdogs", 5: "5: Winners",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    args = parser.parse_args()

    _rows, models = load_rows(args.input)
    all_langs = sorted({lang for m in models.values() for lang in m["token_parity"]})
    levels, unresolved = load_resource_levels(all_langs)
    print(f"n_tokenizers={len(models)}, n_languages={len(all_langs)}, "
          f"{len(unresolved)} unresolved against Joshi's taxonomy (excluded)\n")

    # --- 1. Pooled per-level mean ---
    pooled_by_level = defaultdict(list)
    for m in models.values():
        for lang, p in m["token_parity"].items():
            if lang in levels:
                pooled_by_level[levels[lang]].append(p)

    print("--- pooled mean token_parity per level (matches fig:resource-level-trend's own metric) ---")
    for lvl in sorted(pooled_by_level):
        vals = pooled_by_level[lvl]
        print(f"  {_RESOURCE_LEVEL_LABELS[lvl]:20s} (n={len(vals):4d} points): mean={statistics.mean(vals):.3f}")

    # --- 2 & 3. Pooled correlation + Kruskal-Wallis ---
    pooled_levels, pooled_parities = [], []
    for lvl, vals in pooled_by_level.items():
        pooled_levels.extend([lvl] * len(vals))
        pooled_parities.extend(vals)
    rho, pvalue = stats.spearmanr(pooled_levels, pooled_parities)
    print(f"\npooled Spearman rho={rho:+.3f} (p={pvalue:.2e}, n={len(pooled_levels)})")

    groups = [pooled_by_level[lvl] for lvl in sorted(pooled_by_level)]
    h_stat, kw_pvalue = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H={h_stat:.1f} (p={kw_pvalue:.2e}) across the {len(groups)} resource levels")

    # --- 4. Per-tokenizer consistency ---
    per_tok_rho = {}
    for name, m in models.items():
        lvls, pars = [], []
        for lang, p in m["token_parity"].items():
            if lang in levels:
                lvls.append(levels[lang])
                pars.append(p)
        if len(set(lvls)) < 2:
            continue
        r, p = stats.spearmanr(lvls, pars)
        per_tok_rho[name] = (r, p)

    n_negative_sig = sum(1 for r, p in per_tok_rho.values() if r < 0 and p < 0.05)
    n_positive_sig = sum(1 for r, p in per_tok_rho.values() if r > 0 and p < 0.05)
    n_not_sig = len(per_tok_rho) - n_negative_sig - n_positive_sig
    print(f"\nper-tokenizer trend (level vs. that tokenizer's own per-language parity), n={len(per_tok_rho)} tokenizers:")
    print(f"  {n_negative_sig} show a significant NEGATIVE trend (higher resource -> lower parity, the expected direction)")
    print(f"  {n_positive_sig} show a significant POSITIVE trend (opposite direction)")
    print(f"  {n_not_sig} show no significant trend (p>=0.05)")

    mean_rho = statistics.mean(r for r, _ in per_tok_rho.values())
    print(f"  mean per-tokenizer rho={mean_rho:+.3f}")

    if n_positive_sig:
        print("\n  tokenizers with a significant POSITIVE (opposite-direction) trend:")
        for name, (r, p) in sorted(per_tok_rho.items(), key=lambda kv: -kv[1][0]):
            if r > 0 and p < 0.05:
                print(f"    {name:30s} rho={r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
