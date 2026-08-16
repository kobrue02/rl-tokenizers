"""Overlap-detection logic for checking whether a pretraining corpus shares
text with a downstream eval benchmark's test/devtest examples
(systems.pretraining.benchmarks) -- a known risk since a web-scale crawl
(fineweb_edu/olmo_mix especially) may already contain near-duplicates of a
benchmark's test rows.

APPROACH: the benchmark side is small, so its text is shingled once into an
in-memory index (build_benchmark_shingle_index); the large corpus side is
streamed one document at a time (scan_texts_for_contamination, pure text,
no tokenization/GPU) and checked against that index. This exploits the size
asymmetry instead of indexing the whole corpus. See cli_contamination.py
for the CLI wiring this to a corpus source and benchmark loader.

This is an exact n-gram shingle match, not a MinHash/Jaccard similarity
estimate like common.data.dedup's near-dup detection -- exact matching is
simpler and precise enough given the benchmark side is small and fixed,
and the question is "does any corpus document contain this literal test
text", not overall document similarity.
"""

from collections import defaultdict

from common.data.text_shingles import shingle_hash, shingles


def build_benchmark_shingle_index(examples, text_fields_fn, n=13):
    """examples: a list (indexed by position, so must support len()/repeat
    iteration, not a one-shot generator) of benchmark example objects.
    text_fields_fn(example) -> list[str] pulls the fields to check (e.g.
    XNLI's premise+hypothesis, FLORES-MT's source+reference).

    Returns {shingle_hash: {"text": str, "example_indices": set[int]}};
    example_indices are positions in `examples`."""
    index = {}
    for i, ex in enumerate(examples):
        for field_text in text_fields_fn(ex):
            if not field_text:
                continue
            for s in shingles(field_text, n):
                h = shingle_hash(s)
                entry = index.setdefault(h, {"text": s, "example_indices": set()})
                entry["example_indices"].add(i)
    return index


def scan_texts_for_contamination(text_iter, shingle_index, n=13):
    """text_iter: iterable of raw corpus document strings (one pass; a
    generator is fine here, unlike `examples` above). Returns
    {example_index: set of matched shingle texts} plus the number of
    documents scanned (so a caller that capped the scan knows the
    coverage)."""
    hits = defaultdict(set)
    docs_scanned = 0
    for text in text_iter:
        docs_scanned += 1
        for s in shingles(text, n):
            entry = shingle_index.get(shingle_hash(s))
            if entry is not None:
                for i in entry["example_indices"]:
                    hits[i].add(entry["text"])
    return dict(hits), docs_scanned


def summarize_contamination(examples, hits, docs_scanned):
    """hits: scan_texts_for_contamination's own first return value.
    Returns the JSON-serializable summary systems.pretraining.cli_contamination
    writes out -- factored out so both the CLI and a direct caller (e.g. a
    test) get the exact same report shape."""
    return {
        "num_examples": len(examples),
        "num_contaminated": len(hits),
        "contamination_rate": len(hits) / len(examples) if examples else 0.0,
        "corpus_docs_scanned": docs_scanned,
        "contaminated_examples": [
            {"index": i, "matched_shingles": sorted(shingle_texts)}
            for i, shingle_texts in sorted(hits.items())
        ],
    }
