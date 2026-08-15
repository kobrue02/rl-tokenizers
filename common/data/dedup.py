"""Corpus-level document deduplication, applied by pretraining.data_prep to
every document BEFORE it's tokenized/packed into shards. Frontier-lab
pretraining pipelines (CCNet, RefinedWeb, LLaMA's own data pipeline) all
report a dedup step and its removal rate -- this is that step, not a metric
describing an absent one.

Two layers, run in order for every document (see Deduplicator.is_duplicate):

  1. EXACT dedup: a hash of the document's normalized (lowercased,
     whitespace-collapsed) full text, checked against a BloomFilter of
     every hash seen so far (see BloomFilter's own docstring below and the
     MEMORY section further down for why this isn't a plain set). Catches
     verbatim repeats -- the same page scraped twice, a sentence duplicated
     across two of common.data.corpora's sources. Cheap (one hash + a
     handful of bit lookups) and applied to EVERY document regardless of
     length.

  2. NEAR dedup: MinHash (datasketch) over word shingles (see
     common/data/text_shingles.py) + LSH banding for approximate Jaccard
     similarity, at a tractable cost -- computing true pairwise Jaccard
     similarity between every pair of documents is O(corpus_size^2) and not
     feasible at real corpus scale; LSH (Broder 1997) is the standard way
     around that, and the same technique CCNet/RefinedWeb use at this exact
     scale. Only computed for documents with at least `min_words_for_near_dup`
     words -- MinHash over a handful of words is a noisy signal and exact
     dedup already covers most of the value for very short documents, so
     skipping it there is a real, deliberate cost/benefit call, not an
     oversight.

MEMORY, stated plainly: the LSH index holds one MinHash sketch (num_perm
integers) per document ever indexed, for the lifetime of one prep_dataset
call -- this is a real, currently-unavoidable scaling cost (millions of
documents means millions of sketches held in memory at once), not something
this module pretends is free. num_perm=128 keeps each sketch small (a few
hundred bytes), and near-dup is near-inert on short documents anyway (see
Deduplicator's own docstring), so this stays small in practice for
Glot500-style short-row corpora specifically -- but there is still no
external (disk/database-backed) index here; a genuinely enormous corpus of
long documents would need one.

EXACT-dedup used to have the identical problem in a WORSE form (it applies
to every document regardless of length, unlike near-dup): a real cluster
run confirmed a plain Python set of digests grew to the point of OOM-killing
a job partway through a ~30B-token/hundreds-of-millions-of-document
Glot500 prep run that a smaller ~5B-token run of the same code never hit.
Exact-dedup below is now backed by BloomFilter instead -- fixed memory at
construction time regardless of how many documents are actually seen, at
the cost of a small, tunable false-positive rate (a genuinely new document
occasionally, incorrectly treated as a duplicate and dropped) rather than
unbounded growth.
"""

import array
import hashlib
import math

from datasketch import MinHash, MinHashLSH

from .text_shingles import normalize_words, shingles


class BloomFilter:
    """Fixed-memory approximate set membership (Bloom 1970): may say an
    item was seen before when it wasn't (false positive, at `error_rate`),
    never says an item wasn't seen when it was (no false negatives) --
    exactly the safe direction for a dedup filter, where a false positive
    just drops one extra document and a false negative would let a real
    duplicate through.

    Uses the Kirsch/Mitzenmacher (2008) technique: two independent 64-bit
    halves (h1, h2) of one already-computed wide hash combine into
    num_hashes effective probes via g_i = h1 + i*h2 (mod num_bits), rather
    than computing num_hashes independent hashes per item.
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
    FIRST occurrence of any duplicate cluster is always kept; every later
    one is dropped (is_duplicate returns True for it, without re-indexing
    it -- a duplicate of a duplicate doesn't need its own LSH entry).

    A real, verified (not just assumed) property of shingle_size=13 near-dup
    at default settings: it's effective on LONG documents with a FEW
    scattered edits (a re-crawled web page with a typo fixed still clears
    threshold=0.8 easily), and near-INERT on SHORT documents -- a couple of
    scattered word changes in an 18-word sentence corrupts a large fraction
    of its 13-word windows and stays well below 0.8 Jaccard, so it reads as
    a genuinely different sentence, not a near-duplicate. This matches where
    this technique is actually applied in the literature (full-length
    CommonCrawl-style documents, e.g. this project's fineweb_edu/olmo_mix)
    rather than short individual sentences (oldi_seed/flores_dev/smol/
    glot500 rows) -- exact dedup above already catches verbatim short-row
    repeats regardless of length, and "near-duplicate" is a less meaningful
    concept for a single short sentence differing by one word anyway (it
    may just be a genuinely different sentence)."""

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
        # generously above a 30B-token-scale Glot500 prep run's likely
        # document count (short rows -- plausibly upward of a billion
        # documents to reach 30B tokens) rather than tightly, since
        # undershooting capacity just raises the false-positive rate
        # gradually (never a crash, see BloomFilter's own docstring) while
        # the cost of guessing generously here is a couple more GB, cheap
        # next to the OOM this replaced (see this module's own docstring).

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
        """Two independent 64-bit halves of _exact_key's own digest -- see
        BloomFilter's own docstring for why two hashes are enough."""
        digest = Deduplicator._exact_key(text)
        return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")

    def _minhash(self, text):
        mh = MinHash(num_perm=self._num_perm)
        for s in shingles(text, self._shingle_size):
            mh.update(s.encode("utf-8"))
        return mh

    def is_duplicate(self, text):
        """Returns True if `text` is an exact or near duplicate of a
        document already seen this run (and does NOT index it further);
        otherwise indexes it (both layers) and returns False."""
        h1, h2 = self._exact_hash_pair(text)
        if self._seen_exact.might_contain(h1, h2):
            return True
        self._seen_exact.add(h1, h2)  # recorded regardless of the near-dup
        # path below, so a LATER exact repeat of this same short document
        # is still caught even though it never entered the LSH index.

        words = normalize_words(text)
        if len(words) < self._min_words_for_near_dup:
            return False

        mh = self._minhash(text)
        if self._lsh.query(mh):
            return True
        self._lsh.insert(str(self._next_id), mh)
        self._next_id += 1
        return False
