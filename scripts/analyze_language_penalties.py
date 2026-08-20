"""Answers the Token Tax chapter's \\S{sec:tt-worst} TODO (fig:worst-
language-heatmap): moving from tokenizer-level aggregates to language-level
detail -- which languages are penalised by every tokenizer, which only by
some, does the penalty track script or resource level, and is any
tokenizer anomalously good for a language its own family would not predict?

"Penalised" is defined PER TOKENIZER, not by a fixed absolute threshold:
a language is in tokenizer T's own worst quartile if T's (English-relative)
token_parity for it is at or above T's own 75th percentile across all 259
languages. This is deliberate -- tokenizers differ hugely in their overall
scale of token_parity (see \\S{sec:tt-landscape}), so an absolute cutoff
like "parity > 1.5" would just re-measure each tokenizer's own average
rather than which of ITS languages it penalises most. penalized_count[lang]
= how many tokenizers put that language in their own worst quartile --
259 (or close to it) = penalised near-universally; 0 = never a worst-
quartile language for anyone.

MORPHOLOGY is deliberately NOT analysed here: this project has
morphological-type tags (polysynthetic/agglutinative) only for the
separate 15-language indigenous panel (common.data.indigenous_panel,
different bare-ISO code scheme), not for the 259-language BOUQuET panel --
fabricating a morphology tag for languages this project has no real
annotation for would misrepresent the data. Cross-reference \\S{sec:tt-
indigenous}'s own figures for that narrower, real comparison instead.

ANOMALY DETECTION: for each (language, tokenizer-family) pair with >= 2
tokenizers in that family, a z-score of each member tokenizer's token_parity
against its OWN family's mean/std for that language -- a large negative z
(unusually LOW parity, i.e. unusually GOOD) flags a tokenizer beating its
own family's typical performance for a language that family, as a whole,
doesn't handle well. This directly targets "a tokenizer anomalously good
for a language its family would not predict."

Usage:
    python3 -m scripts.analyze_language_penalties \\
        --input results/all_tokenizers_comparison.json
"""

import argparse
import statistics
from collections import defaultdict

from scipy import stats

from common.data.lang2tax import load_resource_levels
from scripts.generate_tikz_figures import family_of, load_rows

_RESOURCE_LEVEL_LABELS = {
    0: "0: Left-Behinds", 1: "1: Scraping-Bys", 2: "2: Hopefuls",
    3: "3: Rising Stars", 4: "4: Underdogs", 5: "5: Winners",
}


def _script_of(lang_script_code):
    return lang_script_code.split("_", 1)[1] if "_" in lang_script_code else "?"


def _worst_quartile_per_tokenizer(models):
    """{tokenizer_name: set(lang codes in that tokenizer's own worst
    quartile of token_parity)}."""
    worst = {}
    for name, m in models.items():
        parities = m["token_parity"]
        threshold = statistics.quantiles(list(parities.values()), n=4)[-1]  # 75th percentile
        worst[name] = {lang for lang, p in parities.items() if p >= threshold}
    return worst


def _penalized_counts(worst_quartiles, all_langs):
    counts = {lang: 0 for lang in all_langs}
    for langs in worst_quartiles.values():
        for lang in langs:
            counts[lang] += 1
    return counts


def _grouped_means(counts, group_of, min_group_n=3):
    grouped = defaultdict(list)
    for lang, count in counts.items():
        grouped[group_of(lang)].append(count)
    return {
        g: (statistics.mean(vs), len(vs))
        for g, vs in grouped.items()
        if len(vs) >= min_group_n
    }


def _family_anomalies(models, all_langs, top_n=10):
    """For each (lang, family) with >=2 tokenizers, z-score each member's
    token_parity against its own family's mean/std for that language.
    Returns the top_n most negative z-scores (tokenizer unusually GOOD
    relative to its own family, for that language)."""
    by_family_lang = defaultdict(list)  # (family, lang) -> [(tokenizer, parity)]
    for name, m in models.items():
        fam = family_of(name)
        for lang in all_langs:
            if lang in m["token_parity"]:
                by_family_lang[(fam, lang)].append((name, m["token_parity"][lang]))

    anomalies = []
    for (fam, lang), entries in by_family_lang.items():
        if len(entries) < 2:
            continue
        values = [v for _, v in entries]
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        if std == 0:
            continue
        for name, v in entries:
            z = (v - mean) / std
            anomalies.append((z, name, fam, lang, v, mean))
    anomalies.sort(key=lambda a: a[0])  # most negative (best-relative-to-family) first
    return anomalies[:top_n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    rows, models = load_rows(args.input)
    all_langs = sorted({lang for m in models.values() for lang in m["token_parity"]})

    worst_quartiles = _worst_quartile_per_tokenizer(models)
    counts = _penalized_counts(worst_quartiles, all_langs)
    n_tokenizers = len(models)

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"n_tokenizers={n_tokenizers}, n_languages={len(all_langs)}\n")

    print(f"most universally penalised languages (top {args.top_n}):")
    for lang, count in ranked[: args.top_n]:
        print(f"  {lang:15s} in worst quartile for {count}/{n_tokenizers} tokenizers")

    print(f"\nleast penalised languages (top {args.top_n}):")
    for lang, count in ranked[-args.top_n:][::-1]:
        print(f"  {lang:15s} in worst quartile for {count}/{n_tokenizers} tokenizers")

    print("\n--- by script ---")
    by_script = _grouped_means(counts, _script_of, min_group_n=3)
    for script, (mean, n) in sorted(by_script.items(), key=lambda kv: -kv[1][0]):
        print(f"  {script:6s} (n={n:3d} langs): mean penalized_count={mean:.2f}/{n_tokenizers}")

    print("\n--- by Joshi resource level ---")
    levels, unresolved = load_resource_levels(all_langs)
    print(f"({len(unresolved)}/{len(all_langs)} languages not in Joshi's taxonomy, excluded below)")
    by_level = _grouped_means(
        {lang: c for lang, c in counts.items() if lang in levels},
        lambda lang: levels[lang], min_group_n=1,
    )
    for level, (mean, n) in sorted(by_level.items()):
        print(f"  {_RESOURCE_LEVEL_LABELS[level]:20s} (n={n:3d} langs): mean penalized_count={mean:.2f}/{n_tokenizers}")

    resolved_langs = [l for l in all_langs if l in levels]
    level_vals = [levels[l] for l in resolved_langs]
    count_vals = [counts[l] for l in resolved_langs]
    rho, pvalue = stats.spearmanr(level_vals, count_vals)
    print(f"\nresource-level vs. penalized_count Spearman rho={rho:+.3f} (p={pvalue:.4f}, n={len(resolved_langs)})")

    print(f"\n--- family anomalies (tokenizer unusually GOOD vs. its own family), top {args.top_n} ---")
    for z, name, fam, lang, v, mean in _family_anomalies(models, all_langs, top_n=args.top_n):
        print(f"  {name:30s} ({fam:15s}) on {lang:12s}: parity={v:.3f} vs family mean={mean:.3f} (z={z:+.2f})")


if __name__ == "__main__":
    main()
