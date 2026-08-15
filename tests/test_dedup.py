"""BloomFilter correctness (no false negatives, bounded memory) and
Deduplicator's exact/near-dup behavior after exact-dedup moved from an
unbounded set to a fixed-memory BloomFilter -- see common/data/dedup.py's
own module docstring for the real OOM this replaced."""

from common.data.dedup import BloomFilter, Deduplicator


def test_bloom_filter_no_false_negatives():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    pairs = [(i, i * 2 + 1) for i in range(500)]
    for h1, h2 in pairs:
        bf.add(h1, h2)
    for h1, h2 in pairs:
        assert bf.might_contain(h1, h2)


def test_bloom_filter_rejects_most_unseen_items():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    for i in range(500):
        bf.add(i, i * 2 + 1)
    false_positives = sum(
        1 for i in range(500, 5500) if bf.might_contain(i, i * 2 + 1)
    )
    # error_rate=0.01 -> expect roughly 1% false positives among unseen
    # items at design capacity; generous slack since this is a specific
    # random-looking (not adversarial) probe set, not a formal guarantee.
    assert false_positives < 0.05 * 5000


def test_bloom_filter_memory_is_fixed_at_construction():
    bf_small = BloomFilter(capacity=1_000, error_rate=0.01)
    bf_large = BloomFilter(capacity=1_000_000, error_rate=0.01)
    # num_bits scales with capacity (Bloom 1970's own formula), not with
    # however many items actually get added later -- the whole point.
    assert bf_large.num_bits > bf_small.num_bits
    for i in range(2000):  # exceeds bf_small's capacity on purpose
        bf_small.add(i, i * 7 + 3)
    assert len(bf_small._words) == (bf_small.num_bits + 63) // 64


def test_deduplicator_catches_exact_repeat():
    dedup = Deduplicator()
    text = "The quick brown fox jumps over the lazy dog in the morning sun."
    assert dedup.is_duplicate(text) is False
    assert dedup.is_duplicate(text) is True


def test_deduplicator_distinct_documents_not_flagged():
    dedup = Deduplicator()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A completely different sentence about something else entirely.",
        "Yet another unrelated document discussing tokenization fairness.",
    ]
    for text in texts:
        assert dedup.is_duplicate(text) is False


def test_deduplicator_near_duplicate_long_document():
    dedup = Deduplicator(min_words_for_near_dup=10)
    original = " ".join(f"word{i}" for i in range(200))
    near_copy = original.replace("word5", "wordFIVE")  # one scattered edit
    assert dedup.is_duplicate(original) is False
    assert dedup.is_duplicate(near_copy) is True
