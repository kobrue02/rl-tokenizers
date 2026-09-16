"""Fairness/efficiency metrics, implemented once and reused unchanged across every
tokenizer in this repo (fairtok, magnet, flexitokens, manta) so comparisons score
them all the same way.

Rényi efficiency: Zouhar et al., "Tokenization and the Noiseless Channel" (ACL 2023).
Gini coefficient: Foroutan et al., "Parity-aware Byte-Pair Encoding" (2025), Eq. 5
(closed form here derived independently, then checked identical) -- applied to
per-language TOKEN_PARITY (token cost relative to an anchor language), per that
paper's own "per-language token costs" framing, NOT to renyi_efficiency (a
2026-09-16 fix -- see common.eval.cross_tokenizer.evaluate_on_groups's own
docstring for the distinction between the two quantities).
Fertility: Ahia et al., "Do All Languages Cost the Same?" (EMNLP 2023); also the
headline metric of Lundin et al.'s "The Token Tax" (AfricaNLP 2026) -- included for
comparability with that literature, which mostly reports fertility rather than an
entropy-based measure.
Boundary stability: adapted from "Proxy Compression for Language Modeling" (Zheng
et al. 2026), Sec 3.4 -- repurposed here as a per-language fairness check (see
common.eval.stability).
Morphological Edit Distance: Asgari et al., "MorphBPE" (arXiv:2502.00894), Sec 3.3(ii)
-- shares its atomic-sequence edit-distance core with boundary_stability (see
_atomic_sequence_edit_distance); see common.eval.morphology for the full metric
(this module only has the shared DP primitive).
"""

import random

import numpy as np


def renyi_entropy(probs, alpha):
    """probs: 1-D array of probabilities (should sum to ~1, zeros already excluded)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-9:
        return float(-np.sum(probs * np.log(probs)))  # Shannon entropy, limiting case
    return float((1.0 / (1.0 - alpha)) * np.log(np.sum(probs**alpha)))


def renyi_efficiency(freqs, alpha=2.5, vocab_size=None):
    """freqs: raw counts (or probabilities) per token type for one language.

    vocab_size defaults to len(freqs) if omitted -- but that's the number of
    DISTINCT TYPES OBSERVED IN THIS SAMPLE, not the tokenizer's true designed
    vocabulary size. Confirmed live (2026-09-16): every call site in this repo
    omitted vocab_size for months, which cannot detect Zouhar et al.'s target
    failure mode (a vocabulary padded with tokens too rare to ever be learned)
    at all, since an eval sample that happens to use few distinct types looks
    "efficient" regardless of how bloated the tokenizer's real vocabulary is.
    ALWAYS pass the tokenizer's real vocab_size explicitly when it's known;
    the len(freqs) fallback exists only for genuinely vocab-free tokenizers
    (e.g. BLT) where no true vocab_size concept exists at all.
    """
    freqs = np.asarray(list(freqs), dtype=np.float64)
    v = vocab_size if vocab_size is not None else freqs.size
    if v <= 1:
        return 0.0
    total = freqs.sum()
    if total <= 0:
        return 0.0
    probs = freqs[freqs > 0] / total
    return renyi_entropy(probs, alpha) / np.log(v)


def gini_coefficient(values):
    """values: one scalar per language (e.g. per-language token cost or efficiency)."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    total = values.sum()
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * total) / (n * total))


def compression_rate(num_bytes, num_tokens):
    """Bytes represented per token -- higher means fewer tokens for the same content."""
    if num_tokens <= 0:
        return 0.0
    return num_bytes / num_tokens


def fertility(num_tokens, num_words):
    """Tokens per word (higher = less efficient for that language). num_tokens/
    num_words are corpus-level totals, not a per-sentence average, matching how
    the tokenizer-fairness literature reports it."""
    if num_words <= 0:
        return 0.0
    return num_tokens / num_words


def _atomic_sequence_edit_distance(seq_a, seq_b):
    """Plain Levenshtein distance treating each ELEMENT of seq_a/seq_b as one
    atomic symbol (equal iff `==`), not a character/byte-level edit distance --
    a boundary shift inside one long span/morph should count as roughly ONE
    change, not one per byte/character it touches. Shared DP core for
    boundary_stability (below) and common.eval.morphology.morphological_edit_distance,
    which differ only in how they turn this raw distance into a final score
    (1-minus-normalized similarity vs. raw unnormalized distance respectively)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution/match
            )
        prev = curr
    return prev[m]


def boundary_stability(spans_before, spans_after):
    """1 - normalized Levenshtein distance between two span sequences (byte-string
    spans, e.g. from common.bytes_utils.spans_from_boundaries). 1.0 = perturbation
    left the segmentation unchanged, 0.0 = maximally different. See common.eval.stability
    for the perturb-and-compare machinery that produces spans_before/spans_after.

    Treats each span as one atomic symbol (equal iff bytes are identical), not a
    byte-level edit distance -- a boundary shift inside one long span should count
    as roughly ONE change, not one per byte it touches.
    """
    n, m = len(spans_before), len(spans_after)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    edit_distance = _atomic_sequence_edit_distance(spans_before, spans_after)
    return 1.0 - edit_distance / max(n, m)


def bootstrap_ci(outcomes, n_resamples=1000, ci=0.95, seed=0):
    """Percentile bootstrap confidence interval for the MEAN of `outcomes`
    (e.g. a 0/1 correctness list -> accuracy's own CI). Resamples the RAW
    per-example values with replacement n_resamples times -- a genuine
    bootstrap over the underlying trials, not a closed-form approximation
    over pre-aggregated counts -- so the same helper generalizes to any
    per-example statistic (e.g. a paired accuracy DIFFERENCE between two
    systems scored on the same items, which has no simple closed form), not
    just a binomial proportion.

    Moved here from systems.pretraining.eval_harness (2026-09-16) since it's
    a generic statistical primitive common.eval.morphology also needs for
    morphological_consistency_f1's own confidence interval, and common/
    importing from systems/pretraining/ would invert this project's own
    dependency direction (systems/pretraining depends on common/, not vice
    versa). systems.pretraining.eval_harness re-exports this name unchanged,
    so every existing caller/import keeps working without modification.

    Returns (point_estimate, ci_low, ci_high). (0.0, 0.0, 0.0) for empty
    input rather than raising.

    rng.choices(outcomes, k=n) (a single C-level call) rather than a
    Python-level per-element randrange loop -- resampling 1000x at typical
    aggregate eval sizes (thousands of examples) would otherwise dominate
    runtime.
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(outcomes) / n
    rng = random.Random(seed)
    resampled_means = [sum(rng.choices(outcomes, k=n)) / n for _ in range(n_resamples)]
    resampled_means.sort()
    alpha = (1 - ci) / 2
    low_idx = int(alpha * n_resamples)
    high_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples))
    return point, resampled_means[low_idx], resampled_means[high_idx]
