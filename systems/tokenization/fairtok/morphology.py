"""Unsupervised morphological segmentation via Morfessor 2.0, trained on monolingual
text from cis-lmu/Glot500 and allenai/MADLAD-400 -- substitutes for the gold
morphological data (UniMorph/Universal Dependencies) MorphScore normally needs
(arxiv.org/abs/2507.06378), for languages that lack it. (statmt/cc100 was considered
and dropped: no coverage beyond Glot500+MADLAD, plus off-Hub access complexity.)

discover_morph_sources covers every language either source has (389 Glot500 + 303
MADLAD codes, 636 total merged), and merges all available source text per language --
more repeated word-form evidence is what Morfessor needs (a small parallel corpus
alone was too thin/low-repetition).

MERGE RULE: Glot500 uses ISO 639-3-style 3-letter+script codes (e.g. "eng_Latn");
MADLAD mixes ISO 639-1 2-letter codes for higher-resource languages with 3-letter
for the rest. Sources are merged for a language ONLY on an identical code string, never
via a broader 639-1<->639-3 equivalence table -- e.g. MADLAD's generic "ar" is
MSA-dominated Arabic, not necessarily the same variety as some 3-letter Glot500 code,
so guessing them equal risks a real dialect/variety mismatch at a scale no one can
manually vet. _MANUAL_OVERRIDES holds the few cross-code merges (and confirmed
no-data exclusions) that WERE individually checked.

Efficiency: both sources vastly exceed the per-language word budget needed (Morfessor
saturates well under a few million words). Fetchers stream shard-by-shard and stop at
the word budget, so high-resource languages never download their full multi-GB corpus.
"""

import gzip
import json
import re
import unicodedata
from collections import Counter

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

WORD_RE = re.compile(r"\w+", re.UNICODE)

GLOT500_REPO = "cis-lmu/Glot500"
MADLAD_REPO = "allenai/MADLAD-400"

# (source, source_language_code) merges/exclusions needing a one-at-a-time judgment
# call, not the general exact-code-match rule below -- these WIN outright over
# discover_morph_sources's automatic merge for the same code. Determined by querying
# both repos' file trees directly (Glot500's card metadata was incomplete). Notably:
# MADLAD's generic "ar" is NOT used as a stand-in for arz (Egyptian Arabic) -- it's
# MSA-dominated, a real variety mismatch, not just missing data. eng/ben/spa/bam are
# spelled out explicitly since the exact-match rule alone misses their MADLAD 2-letter
# equivalents ("en"/"bn"/"es"/"bm" -- different string, same language).
_MANUAL_OVERRIDES = {
    "arz": [("glot500", "arz_Arab")],
    "bam": [("glot500", "bam_Latn"), ("madlad400", "bm")],
    "ben": [("madlad400", "bn")],
    "eng": [("glot500", "eng_Latn"), ("madlad400", "en")],
    "kas": [],  # no source has any text for this -- confirmed, not a missing lookup
    "lij": [("glot500", "lij_Latn")],
    "mni": [("madlad400", "mni")],
    "nqo": [],  # no source has any text for this -- confirmed, not a missing lookup
    "spa": [("glot500", "spa_Latn"), ("madlad400", "es")],
}


def list_madlad_langs():
    """Every language code allenai/MADLAD-400 offers, discovered from its top-level
    "data-v1p5/" directory listing (non-recursive: listing every file under
    data-v1p5/ takes minutes given ~280k files repo-wide; per-language directories
    alone list fast). Same "ask the source, don't hardcode a copy" convention as
    common.data.corpora.list_glot500_configs. Codes are a mix of ISO 639-1 (2-letter)
    and 639-3 (3-letter) -- MADLAD doesn't use one convention uniformly."""
    tree = HfApi().list_repo_tree(MADLAD_REPO, path_in_repo="data-v1p5", repo_type="dataset")
    return sorted(item.path.rsplit("/", 1)[-1] for item in tree)


def discover_morph_sources(_cache={}):
    """Every language with usable monolingual text in EITHER Glot500 or MADLAD-400,
    discovered live and merged per the module docstring's MERGE RULE (exact
    code-string match, plus _MANUAL_OVERRIDES for checked exceptions). Returns
    {lang_code: [(source_name, source_code), ...]}.

    One canonical script is picked per Glot500 short code when it offers more than one
    (21 of 389 do): reuses common.data.oldi_data.LANG_SCRIPT's vetted choice when
    available (e.g. kas_Arab over kas_Deva), else the alphabetically-first variant
    (arbitrary but deterministic tiebreak).

    Memoized (module-level _cache) -- building this hits the network twice (Glot500
    card_data, MADLAD directory tree). Pass an explicit `{}` first-arg to force
    re-discovery.
    """
    if _cache:
        return _cache
    from common.data.corpora import list_glot500_configs
    from common.data.oldi_data import LANG_SCRIPT

    glot500_by_code = {}
    for stem in list_glot500_configs():
        code = stem.split("_", 1)[0]
        glot500_by_code.setdefault(code, []).append(stem)

    def canonical_stem(code, stems):
        vetted = LANG_SCRIPT.get(code)
        return vetted if vetted in stems else sorted(stems)[0]

    sources = {code: [("glot500", canonical_stem(code, stems))] for code, stems in glot500_by_code.items()}

    for code in list_madlad_langs():
        sources.setdefault(code, []).append(("madlad400", code))

    sources.update(_MANUAL_OVERRIDES)
    _cache.update(sources)
    return _cache


def _words_from_text(text):
    # NFC normalization first: decomposed vs. composed Unicode forms of the same
    # character (common in Arabic-script/diacritic-heavy text) would otherwise
    # count as different word "types" for the same word.
    return WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def _iter_glot500_words(stem, max_words, _file_cache={}):
    # _file_cache={} is a deliberate mutable-default-argument cross-call cache --
    # without it, every language re-lists Glot500's full ~1800-file tree from scratch.
    if GLOT500_REPO not in _file_cache:
        _file_cache[GLOT500_REPO] = list_repo_files(GLOT500_REPO, repo_type="dataset")
    files = sorted(
        f
        for f in _file_cache[GLOT500_REPO]
        if f.startswith(f"{stem}/train/") and f.endswith(".arrow")
    )

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
    # Per-language directory listing (recursive under "data-v1p5/{code}"), not a
    # repo-wide list_repo_files -- the repo has ~284k files across ~300 languages,
    # so listing everything just to filter to one language is an avoidable cost
    # at the hundreds-of-languages scale discover_morph_sources runs at.
    if code not in _file_cache:
        tree = HfApi().list_repo_tree(MADLAD_REPO, path_in_repo=f"data-v1p5/{code}", repo_type="dataset", recursive=True)
        _file_cache[code] = [item.path for item in tree]
    files = _file_cache[code]
    # prefer clean_docs (filtered) shards; only fall through to noisy_docs if clean
    # doesn't cover the word budget on its own
    ordered = sorted(f for f in files if "clean_docs" in f) + sorted(
        f for f in files if "noisy_docs" in f
    )

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


def collect_word_counts(
    lang, max_words_per_source=2_000_000, max_word_types=300_000, sources=None
):
    """sources defaults to discover_morph_sources()[lang]; pass explicitly to override.
    Returns an empty Counter (never raises) if lang has no usable source (e.g.
    kas/nqo, see _MANUAL_OVERRIDES)."""
    sources = discover_morph_sources().get(lang, []) if sources is None else sources
    counts = Counter()
    for source_name, code in sources:
        counts.update(_FETCHERS[source_name](code, max_words_per_source))
    if max_word_types and len(counts) > max_word_types:
        # keep the most frequent types -- Morfessor's cost scales with lexicon size,
        # so this bounds compute for high-resource languages only.
        counts = Counter(dict(counts.most_common(max_word_types)))
    return counts


def train_morfessor(
    word_counts, freq_threshold=1, corpusweight=1.0, init_rand_split=0.5, seed=0
):
    """word_counts: a Counter/dict of {word: count} (see collect_word_counts).
    freq_threshold discards word types occurring fewer than this many times
    (crawl-noise/typo filtering). Returns None if word_counts is empty.

    init_rand_split=0.5 is NOT cosmetic: BaselineModel's default 'recursive'
    algorithm starts every word fully unsplit and, without random seed splits,
    converges right back to zero splits even on textbook cases -- the local search
    never considers a split if none exists to compare against. A nonzero
    init_rand_split seeds candidate split points so it can actually find real ones
    (e.g. 'unhappiness' -> ['un', 'happiness']).

    corpusweight is a best-guess setting from limited manual probing, not properly
    validated -- Morfessor's recommended tuning method (against a hand-checked dev
    set of known-correct segmentations) hasn't been done here."""
    if not word_counts:
        return None
    import random

    import morfessor

    random.seed(seed)  # train_batch optimizes compounds "in a random order" per its
    # own docstring -- this is what makes that reproducible
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data(
        [(count, word) for word, count in word_counts.items()],
        freqthreshold=freq_threshold,
        init_rand_split=init_rand_split,
    )
    model.train_batch(algorithm="recursive")
    return model
