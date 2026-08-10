"""Unsupervised morphological segmentation via Morfessor 2.0, trained on monolingual
text pulled from cis-lmu/Glot500 and allenai/MADLAD-400 -- the substitute for gold
morphological data (UniMorph/Universal Dependencies) that MorphScore normally needs
(arxiv.org/abs/2507.06378), for the languages that don't have any (see conversation:
of our 9-language panel, only eng/spa have UD coverage; Glot500+MADLAD together get 7
of 9 -- kas and nqo have no data in either).

statmt/cc100 was considered and dropped: it contributes zero languages not already
covered by Glot500 or MADLAD, and its data lives off-Hub (data.statmt.org via a legacy
`datasets` loading script, not fetchable via hf_hub_download), so it added real access
complexity for no coverage benefit.

Per-language source list (see MORPH_SOURCES below) merges every available source's
text -- e.g. eng/spa pull from both Glot500 and MADLAD, not just one, since more
repeated word-form evidence is exactly what Morfessor needs (see conversation: the
project's own OLDI-and-friends corpus alone was found too thin/low-repetition for it).
kas and nqo map to an empty source list -- callers must handle that (no model trained,
not an error).

Efficiency: both sources are far larger than any per-language word budget we need
(Morfessor saturates well before billions of words; low hundreds of thousands to a few
million is already a big improvement over what this project's own ~150K-word corpora
offered). Every fetcher streams shard-by-shard and stops as soon as the word budget is
hit, so a high-resource language (spa) never downloads more than its first shard or
two, instead of the full multi-GB corpus.
"""

import gzip
import json
import re
import unicodedata
from collections import Counter

from huggingface_hub import hf_hub_download, list_repo_files

WORD_RE = re.compile(r"\w+", re.UNICODE)

GLOT500_REPO = "cis-lmu/Glot500"
MADLAD_REPO = "allenai/MADLAD-400"

# (source, source_language_code) pairs to merge, per our own language code. Determined
# by directly querying both repos' file trees (not just card metadata, which was
# incomplete for Glot500) -- see conversation for the full derivation. MADLAD's generic
# "ar" is deliberately NOT used as a stand-in for arz (Egyptian Arabic): it's
# MSA-dominated, not the dialect, so using it would be a real variety mismatch, not
# just missing data.
MORPH_SOURCES = {
    "arz": [("glot500", "arz_Arab")],
    "bam": [("glot500", "bam_Latn"), ("madlad400", "bm")],
    "ben": [("madlad400", "bn")],
    "eng": [("glot500", "eng_Latn"), ("madlad400", "en")],
    "kas": [],  # no source has ANY text for this -- see module docstring
    "lij": [("glot500", "lij_Latn")],
    "mni": [("madlad400", "mni")],
    "nqo": [],  # no source has ANY text for this -- see module docstring
    "spa": [("glot500", "spa_Latn"), ("madlad400", "es")],
}


def _words_from_text(text):
    # NFC normalization first: decomposed vs. composed Unicode forms of the same
    # character (common in Arabic-script/diacritic-heavy text especially) would
    # otherwise count as different word "types" for the exact same word.
    return WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def _iter_glot500_words(stem, max_words, _file_cache={}):
    # _file_cache={} is an intentional cross-call cache (the classic Python mutable-
    # default-argument gotcha, used deliberately here) -- without it, every one of the
    # ~7 languages we train would re-list Glot500's full ~1800-file tree from scratch.
    if GLOT500_REPO not in _file_cache:
        _file_cache[GLOT500_REPO] = list_repo_files(GLOT500_REPO, repo_type="dataset")
    files = sorted(f for f in _file_cache[GLOT500_REPO] if f.startswith(f"{stem}/train/") and f.endswith(".arrow"))

    import pyarrow as pa
    import pyarrow.ipc as ipc

    total = 0
    for f in files:
        path = hf_hub_download(GLOT500_REPO, f, repo_type="dataset")
        with pa.memory_map(path, "r") as source:
            try:
                reader = ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
            table = reader.read_all()
        for text in table.column("text").to_pylist():
            words = _words_from_text(text)
            total += len(words)
            yield from words
            if total >= max_words:
                return


def _iter_madlad_words(code, max_words, _file_cache={}):
    if MADLAD_REPO not in _file_cache:
        _file_cache[MADLAD_REPO] = list_repo_files(MADLAD_REPO, repo_type="dataset")
    files = [f for f in _file_cache[MADLAD_REPO] if f.startswith(f"data-v1p5/{code}/")]
    # prefer clean_docs (filtered) shards; only fall through to noisy_docs if clean
    # doesn't cover the word budget on its own
    ordered = sorted(f for f in files if "clean_docs" in f) + sorted(f for f in files if "noisy_docs" in f)

    total = 0
    for f in ordered:
        path = hf_hub_download(MADLAD_REPO, f, repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                words = _words_from_text(row["text"])
                total += len(words)
                yield from words
                if total >= max_words:
                    return


_FETCHERS = {"glot500": _iter_glot500_words, "madlad400": _iter_madlad_words}


def collect_word_counts(lang, max_words_per_source=2_000_000, max_word_types=300_000, sources=None):
    """sources defaults to MORPH_SOURCES[lang]; pass explicitly to override (e.g. for
    a language not in the built-in registry). Returns a Counter, or an empty Counter
    if lang has no configured source (e.g. kas/nqo) -- callers check for emptiness,
    this never raises for a known-uncovered language."""
    sources = MORPH_SOURCES.get(lang, []) if sources is None else sources
    counts = Counter()
    for source_name, code in sources:
        counts.update(_FETCHERS[source_name](code, max_words_per_source))
    if max_word_types and len(counts) > max_word_types:
        # keep the most frequent types -- Morfessor's training cost scales with
        # lexicon size more than raw token count, so this bounds compute for
        # high-resource languages without touching the (already-fine) low-resource ones
        counts = Counter(dict(counts.most_common(max_word_types)))
    return counts


def train_morfessor(word_counts, freq_threshold=1, corpusweight=1.0, init_rand_split=0.5, seed=0):
    """word_counts: a Counter (or dict) of {word: count} -- see collect_word_counts.
    freq_threshold is forwarded to Morfessor's own load_data (discard word types
    occurring fewer than this many times -- crawl noise/OCR-error/singleton-typo
    filtering). Returns None if word_counts is empty (nothing to train on).

    init_rand_split=0.5 is NOT a cosmetic default -- without it, BaselineModel's
    default 'recursive' training algorithm starts every word fully unsplit and,
    empirically (see conversation), converges right back to zero splits even on
    textbook cases (many distinct stems sharing common suffixes, both synthetic and
    real): the recursive local search has no way to discover that splitting would
    lower cost if it never considers a split in the first place. Passing a nonzero
    init_rand_split seeds every word with random candidate split points so the
    search has something to compare against and actually finds real splits
    (e.g. 'unhappiness' -> ['un', 'happiness']).

    That said: this is the best setting found through fairly limited manual probing,
    not a properly validated one. Behavior was noticeably sensitive to word-count
    scale in ways that weren't monotonically "more data = better" in quick testing --
    Morfessor's own recommended way to pick corpusweight is tuning it against a small
    hand-checked development set of known-correct segmentations (see
    BaselineModel.set_corpus_weight_updater / the annotation-based tuning support in
    the morfessor package), which hasn't been done here yet. Treat this as a
    reasonable starting point to iterate from, not a validated final setting."""
    if not word_counts:
        return None
    import random

    import morfessor

    random.seed(seed)  # train_batch optimizes compounds "in a random order" per its
    # own docstring -- seeding Python's random module is what makes that reproducible
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data(
        [(count, word) for word, count in word_counts.items()],
        freqthreshold=freq_threshold, init_rand_split=init_rand_split,
    )
    model.train_batch(algorithm="recursive")
    return model
