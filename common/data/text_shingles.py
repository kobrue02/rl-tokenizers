"""Word n-gram shingle extraction, shared by common.data.dedup (near-
duplicate detection within a corpus) and systems.pretraining.contamination (overlap
between a corpus and an eval benchmark) -- the same primitive, used two
ways: dedup compares corpus documents against each other (both sides large,
needs MinHash/LSH), contamination compares against a small, fixed benchmark
set (fits in memory directly).
"""

import hashlib
import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_SHINGLE_SIZE = 13  # word n-gram length -- matches the GPT-3/PaLM
# contamination-report convention (13-gram overlap): long enough that a
# shared span is a genuine signal, short enough that short documents still
# contribute several shingles.


def normalize_words(text):
    """Lowercased, punctuation-stripped word list -- shingles are computed
    over this, not the raw string, so documents differing only in
    capitalization/punctuation still shingle identically."""
    return _WORD_RE.findall(text.lower())


def shingles(text, n=DEFAULT_SHINGLE_SIZE):
    """Word n-gram shingles of `text` as plain strings. A document shorter
    than `n` words yields ONE shingle (its whole normalized text) rather
    than none, so short documents still get a (coarser) signal."""
    words = normalize_words(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def shingle_hash(shingle_text):
    """A fixed-size (64-bit) int hash of one shingle, for storing/comparing
    at corpus scale (hashes cost far less memory than raw strings). Uses
    blake2b, not builtin hash() -- hash() is salted per-process
    (PYTHONHASHSEED), and the same text must hash identically across runs
    to work as a stored index."""
    return int.from_bytes(hashlib.blake2b(shingle_text.encode("utf-8"), digest_size=8).digest(), "big")
