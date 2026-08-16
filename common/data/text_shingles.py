"""Word n-gram shingle extraction, shared by common.data.dedup (near-duplicate
detection WITHIN a pretraining corpus) and pretraining.contamination
(overlap BETWEEN a pretraining corpus and an eval benchmark's own text) --
the same underlying primitive ("does this text share unusual substrings
with that one"), used two different ways: dedup compares corpus documents
against each other (both sides large, needs MinHash/LSH -- see
common/data/dedup.py), contamination compares corpus documents against a small,
fixed benchmark set (small side fits in memory directly -- see
pretraining/contamination.py).
"""

import hashlib
import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_SHINGLE_SIZE = 13  # word n-gram length -- matches the convention
# GPT-3/PaLM-style contamination reports use (13-gram overlap): long enough
# that a shared span is a genuine signal rather than a common short phrase
# colliding by chance, short enough that even moderately short documents
# still contribute several shingles.


def normalize_words(text):
    """Lowercased, punctuation-stripped word list -- shingles are computed
    over THIS, not the raw string, so two documents differing only in
    capitalization/punctuation still shingle identically."""
    return _WORD_RE.findall(text.lower())


def shingles(text, n=DEFAULT_SHINGLE_SIZE):
    """Word n-gram shingles of `text` as plain strings (space-joined words).
    A document shorter than `n` words yields ONE shingle (its whole
    normalized text) rather than none -- short documents still deserve a
    dedup/contamination signal, just a coarser one."""
    words = normalize_words(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def shingle_hash(shingle_text):
    """A fixed-size (64-bit) int hash of one shingle -- used wherever
    shingles need to be stored/compared at real corpus scale (millions of
    distinct raw shingle strings would cost far more memory than their
    hashes). blake2b (not Python's builtin hash()) because builtin hash()
    is salted per-process by default (PYTHONHASHSEED) -- two runs of the
    SAME text must hash identically for this to mean anything as a stored
    index, not just within one process's lifetime."""
    return int.from_bytes(hashlib.blake2b(shingle_text.encode("utf-8"), digest_size=8).digest(), "big")
