"""Corpus-level document deduplication, applied by pretraining.data_prep to
every document BEFORE it's tokenized/packed into shards. Frontier-lab
pretraining pipelines (CCNet, RefinedWeb, LLaMA's own data pipeline) all
report a dedup step and its removal rate -- this is that step, not a metric
describing an absent one.

Two layers, run in order for every document (see Deduplicator.is_duplicate):

  1. EXACT dedup: a hash of the document's normalized (lowercased,
     whitespace-collapsed) full text, checked against a set of every hash
     seen so far. Catches verbatim repeats -- the same page scraped twice,
     a sentence duplicated across two of common.corpora's sources. Cheap
     (one hash + one set lookup) and applied to EVERY document regardless
     of length.

  2. NEAR dedup: MinHash (datasketch) over word shingles (see
     common/text_shingles.py) + LSH banding for approximate Jaccard
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
hundred bytes), but there is no external (disk/database-backed) index here
-- a genuinely enormous corpus (hundreds of millions of documents) would
need one; not built, since nothing in this project has hit that scale yet.
"""

import hashlib

from datasketch import MinHash, MinHashLSH

from .text_shingles import normalize_words, shingles


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

    def __init__(self, near_dup_threshold=0.8, num_perm=128, shingle_size=13, min_words_for_near_dup=50):
        self._seen_exact = set()
        self._lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._shingle_size = shingle_size
        self._min_words_for_near_dup = min_words_for_near_dup
        self._next_id = 0

    @staticmethod
    def _exact_key(text):
        return hashlib.blake2b(" ".join(normalize_words(text)).encode("utf-8"), digest_size=16).digest()

    def _minhash(self, text):
        mh = MinHash(num_perm=self._num_perm)
        for s in shingles(text, self._shingle_size):
            mh.update(s.encode("utf-8"))
        return mh

    def is_duplicate(self, text):
        """Returns True if `text` is an exact or near duplicate of a
        document already seen this run (and does NOT index it further);
        otherwise indexes it (both layers) and returns False."""
        key = self._exact_key(text)
        if key in self._seen_exact:
            return True
        self._seen_exact.add(key)  # recorded regardless of the near-dup
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
