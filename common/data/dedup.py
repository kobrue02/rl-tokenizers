"""Corpus-level document deduplication, applied by systems.pretraining.data_prep to
every document before it's tokenized/packed into shards (the same kind of
dedup step CCNet/RefinedWeb/LLaMA's own pipelines report).

Two layers, run in order for every document (see Deduplicator.is_duplicate):

  1. EXACT dedup: hash of the normalized (lowercased, whitespace-collapsed)
     full text, checked against a BloomFilter of every hash seen so far
     (see BloomFilter and the MEMORY note below for why not a plain set).
     Catches verbatim repeats -- the same page scraped twice, a sentence
     duplicated across common.data.corpora sources. Applied to EVERY
     document regardless of length.

  2. NEAR dedup: MinHash (datasketch) over word shingles
     (common/data/text_shingles.py) + LSH banding for approximate Jaccard
     similarity -- true pairwise Jaccard is O(corpus_size^2) and infeasible
     at real scale; LSH (Broder 1997) is the standard fix, the same
     technique CCNet/RefinedWeb use at similar scale. Only computed for
     documents with >= `min_words_for_near_dup` words -- MinHash over a
     handful of words is noisy, and exact dedup already covers most short
     documents.

MEMORY: the LSH index holds one MinHash sketch (num_perm integers) per
document ever indexed, for the lifetime of one prep_dataset call --
millions of documents means millions of sketches held in memory at once.
num_perm=128 keeps each sketch small (a few hundred bytes), and near-dup is
near-inert on short documents anyway (see Deduplicator), so this stays
manageable for Glot500-style short-row corpora -- but there is no external
(disk/database-backed) index here; a corpus of long documents would need
one.

Exact-dedup used to have the same problem in a WORSE form (it applies
regardless of length): a real cluster run confirmed a plain Python set of
digests OOM-killed a job partway through a ~30B-token Glot500 prep run.
It's now backed by BloomFilter instead -- fixed memory at construction time
regardless of how many documents are seen, at the cost of a small, tunable
false-positive rate rather than unbounded growth.
"""

import array
import hashlib
import math

from datasketch import MinHash, MinHashLSH

from .text_shingles import normalize_words, shingles


class BloomFilter:
    """Fixed-memory approximate set membership (Bloom 1970): may say an item
    was seen before when it wasn't (false positive, at `error_rate`), never
    says an item wasn't seen when it was -- the safe direction for a dedup
    filter, where a false positive just drops one extra document.

    Uses the Kirsch/Mitzenmacher (2008) technique: two independent 64-bit
    halves (h1, h2) of one already-computed hash combine into num_hashes
    probes via g_i = h1 + i*h2 (mod num_bits), avoiding num_hashes
    independent hashes per item.
    """

    def __init__(self, capacity, error_rate=0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        # standard optimal-parameter formulas (Bloom 1970):
        # num_bits = -capacity*ln(error_rate) / ln(2)^2,
        # num_hashes = (num_bits/capacity)*ln(2).
        self.num_bits = max(8, int(-capacity * math.log(error_rate) / (math.log(2) ** 2)))
        self.num_hashes = max(1, round((self.num_bits / capacity) * math.log(2)))
        self._words = array.array("Q", [0]) * ((self.num_bits + 63) // 64)

    def _positions(self, h1, h2):
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, h1, h2):
        for pos in self._positions(h1, h2):
            self._words[pos // 64] |= 1 << (pos % 64)

    def might_contain(self, h1, h2):
        return all(self._words[pos // 64] & (1 << (pos % 64)) for pos in self._positions(h1, h2))


class Deduplicator:
    """One instance covers ONE prep_dataset call -- documents are compared
    against every document already seen IN THIS RUN, in corpus order. The
    FIRST occurrence of a duplicate cluster is kept; later ones are dropped
    without re-indexing.

    At default settings (shingle_size=13), near-dup is effective on LONG
    documents with a FEW scattered edits (a re-crawled page with a typo
    fixed still clears threshold=0.8), and near-INERT on SHORT documents (a
    couple of word changes in an 18-word sentence corrupts too many of its
    13-word windows to clear 0.8 Jaccard) -- matching where this technique
    is meant to apply (full-length documents like fineweb_edu/olmo_mix)
    rather than short sentences (oldi_seed/flores_dev/smol/glot500 rows),
    which exact dedup above already covers."""

    def __init__(
        self,
        near_dup_threshold=0.8,
        num_perm=128,
        shingle_size=13,
        min_words_for_near_dup=50,
        exact_dedup_capacity=2_000_000_000,
        exact_dedup_error_rate=0.01,
    ):
        # capacity/error_rate defaults: ~2.4GB fixed at construction. Sized
        # generously above a 30B-token-scale Glot500 run's likely document
        # count, since undershooting just raises the false-positive rate
        # gradually (never a crash) -- a couple extra GB is cheap next to
        # the OOM this replaced (see module docstring).

        self._seen_exact = BloomFilter(exact_dedup_capacity, exact_dedup_error_rate)
        self._lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._shingle_size = shingle_size
        self._min_words_for_near_dup = min_words_for_near_dup
        self._next_id = 0

    @staticmethod
    def _exact_key(text):
        return hashlib.blake2b(" ".join(normalize_words(text)).encode("utf-8"), digest_size=16).digest()

    @staticmethod
    def _exact_hash_pair(text):
        """Two independent 64-bit halves of _exact_key's digest -- see
        BloomFilter for why two hashes suffice."""
        digest = Deduplicator._exact_key(text)
        return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")

    def _minhash(self, text):
        mh = MinHash(num_perm=self._num_perm)
        for s in shingles(text, self._shingle_size):
            mh.update(s.encode("utf-8"))
        return mh

    def is_duplicate(self, text):
        """True if `text` is an exact or near duplicate of a document
        already seen this run (without indexing it further); otherwise
        indexes it (both layers) and returns False."""
        h1, h2 = self._exact_hash_pair(text)
        if self._seen_exact.might_contain(h1, h2):
            return True
        self._seen_exact.add(h1, h2)  # recorded even on the near-dup path
        # below, so a later exact repeat is still caught even if it never
        # entered the LSH index.

        words = normalize_words(text)
        if len(words) < self._min_words_for_near_dup:
            return False

        mh = self._minhash(text)
        if self._lsh.query(mh):
            return True
        self._lsh.insert(str(self._next_id), mh)
        self._next_id += 1
        return False
