"""Core overlap-detection logic for checking whether a pretraining corpus
shares text with a downstream eval benchmark's own test/devtest examples
(pretraining.benchmarks) -- a real, known risk in the literature (a
generic web-scale crawl, especially this project's fineweb_edu/olmo_mix,
may already contain near-duplicates of a benchmark's own test rows,
independent of anything this project's own tokenizer/pretraining pipeline
did) that pretraining.benchmarks's own module docstring flags as an open
risk this module exists to actually check, not just describe.

APPROACH: the benchmark side is small (hundreds-thousands of examples) --
its own text is shingled ONCE into an in-memory index (see
build_benchmark_shingle_index). The corpus side is large; scan_texts_for_
contamination streams it ONE DOCUMENT AT A TIME (no tokenization, no GPU --
a pure-text pass, far cheaper than pretraining.data_prep's own tokenization
step) and checks each document's shingles against that small index. This
exploits the size asymmetry (index the SMALL side, stream-scan the LARGE
side) instead of trying to index the entire pretraining corpus, which would
need far more memory for no benefit here -- see pretraining/cli_contamination.py
for the CLI that wires this to an actual common.corpora source and a
pretraining.benchmarks loader.

This is a TEXT-level exact n-gram MATCH (a shingle either appears in the
index or it doesn't), not a MinHash/Jaccard SIMILARITY estimate like
common.dedup's near-dup detection -- the benchmark side is fixed and small
enough that exact per-shingle indexing is both simpler and more precise
(no approximation), and what this is actually looking for is "does ANY
corpus document contain this literal test example's text", not "how
similar are two whole documents overall."
"""

from collections import defaultdict

from common.text_shingles import shingle_hash, shingles


def build_benchmark_shingle_index(examples, text_fields_fn, n=13):
    """examples: a list (not a one-shot generator -- indexed by position
    below, so this must support len()/repeat iteration) of benchmark
    example objects (pretraining.benchmarks.MultipleChoiceExample/
    TranslationExample). text_fields_fn(example) -> list[str] pulls out
    whichever of that example's own fields should be checked (e.g. XNLI's
    premise+hypothesis, FLORES-MT's source+reference text).

    Returns {shingle_hash: {"text": str, "example_indices": set[int]}} --
    example_indices are POSITIONS in `examples` (the caller keeps that same
    list around to look a hit back up against the original example)."""
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
    """text_iter: iterable of raw corpus document strings (ONE PASS --
    a generator is fine here, unlike build_benchmark_shingle_index's
    `examples`). Returns {example_index: set of matched shingle texts} for
    every benchmark example that shares at least one n-gram with the
    corpus documents seen. Also returns the number of documents scanned
    (as a 2-tuple), since a caller that capped the scan needs that number
    to know what the result actually covers."""
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
    Returns the JSON-serializable summary pretraining.cli_contamination
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
