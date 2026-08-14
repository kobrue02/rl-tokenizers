"""Fairness/efficiency metrics, implemented once and reused unchanged across every
tokenizer in this repo (fairtok's RL policy, and the magnet/flexitokens/manta
baselines) -- comparing them meaningfully requires scoring them all the same way.

Rényi efficiency: Zouhar et al., "Tokenization and the Noiseless Channel" (ACL 2023).
Gini coefficient: Foroutan et al., "Parity-aware Byte-Pair Encoding" (2025), Eq. 5 -- the
closed form here is algebraically identical to theirs (derived independently, then checked).
Fertility: Ahia et al., "Do All Languages Cost the Same?" (EMNLP 2023); also the
headline metric of Lundin et al.'s "The Token Tax" (AfricaNLP 2026) -- included
alongside Rényi efficiency/Gini for comparability with that wider literature, most of
which reports fertility rather than an entropy-based measure.
Boundary stability: adapted from "Proxy Compression for Language Modeling" (Zheng et
al. 2026)'s "compressor stability" diagnostic (Sec 3.4) -- there used to explain why
gzip-compressed training proxies fail to transfer while tokenizer/neural-compressor
proxies succeed; repurposed here as a per-language fairness check (see common.eval.stability).
"""

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

    vocab_size defaults to len(freqs) -- i.e. the number of vocabulary slots the
    caller is normalizing against, INCLUDING zero-frequency entries. Pass it
    explicitly if `freqs` has already been filtered to nonzero entries only.
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
    """Tokens per word -- see module docstring. Higher = more tokens needed per word =
    a language this tokenizer serves less efficiently. num_tokens/num_words are
    corpus-level totals (summed over every sentence for one language), not a
    per-sentence average, matching how the tokenizer-fairness literature reports it."""
    if num_words <= 0:
        return 0.0
    return num_tokens / num_words


def boundary_stability(spans_before, spans_after):
    """1 - normalized Levenshtein distance between two span sequences (each a list of
    byte-string spans, e.g. from common.bytes_utils.spans_from_boundaries) -- 1.0 means
    an input perturbation left the induced segmentation completely unchanged, 0.0 means
    maximally different. See module docstring for where this is adapted from, and
    common.eval.stability for the perturb-and-compare machinery that produces
    spans_before/spans_after in the first place.

    Treats each span as one atomic symbol (two spans are "equal" iff their bytes are
    identical), not a byte-level edit distance -- a single boundary shift several bytes
    into a long span should count as roughly ONE segmentation change, not one change
    per byte it happens to touch.
    """
    n, m = len(spans_before), len(spans_after)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if spans_before[i - 1] == spans_after[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution/match
            )
        prev = curr
    edit_distance = prev[m]
    return 1.0 - edit_distance / max(n, m)
