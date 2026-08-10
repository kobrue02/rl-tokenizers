"""Fairness/efficiency metrics, implemented once and reused unchanged across every
tokenizer in this repo (fairtok's RL policy, and the magnet/flexitokens/manta
baselines) -- comparing them meaningfully requires scoring them all the same way.

Rényi efficiency: Zouhar et al., "Tokenization and the Noiseless Channel" (ACL 2023).
Gini coefficient: Foroutan et al., "Parity-aware Byte-Pair Encoding" (2025), Eq. 5 -- the
closed form here is algebraically identical to theirs (derived independently, then checked).
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
