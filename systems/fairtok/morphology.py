"""Unsupervised morphological segmentation via Morfessor 2.0, trained on monolingual
text pulled from cis-lmu/Glot500 and allenai/MADLAD-400 -- the substitute for gold
morphological data (UniMorph/Universal Dependencies) that MorphScore normally needs
(arxiv.org/abs/2507.06378), for languages that don't have any.

statmt/cc100 was considered and dropped: it contributes zero languages not already
covered by Glot500 or MADLAD, and its data lives off-Hub (data.statmt.org via a legacy
`datasets` loading script, not fetchable via hf_hub_download), so it added real access
complexity for no coverage benefit.

Per-language source list (see discover_morph_sources below) covers EVERY language
either source has, not a hand-picked handful -- confirmed live: 389 unique Glot500
short codes + 303 MADLAD-400 codes, 636 total once merged. Merges every available
source's text for a given language -- more repeated word-form evidence is exactly
what Morfessor needs (see conversation: a small parallel corpus alone was found too
thin/low-repetition for it).

MERGE RULE, and why it stops where it does: Glot500 and MADLAD-400 don't share one
code convention (Glot500 uses ISO 639-3-style 3-letter codes with a script suffix,
e.g. "eng_Latn"; MADLAD mixes ISO 639-1 2-letter codes for higher-resource languages,
e.g. "en", with 3-letter codes for the rest) -- discover_morph_sources merges the two
sources for a language ONLY when their own codes are the IDENTICAL string, never via
a broader ISO 639-1<->639-3 equivalence table. That's a deliberate, conservative
boundary, not a missed opportunity: MADLAD's generic "ar" is MSA-dominated Arabic, not
any one specific dialect/variety a 3-letter code might name, so silently equating
"ar" with some 3-letter Glot500 variety code purely because they're "probably the same
language" risks exactly the kind of variety mismatch a real linguist would catch and a
blind code-mapping table wouldn't -- safer to keep them as two separate entries than
guess wrong at scale across hundreds of languages with no way to manually vet each one.
_MANUAL_OVERRIDES below is where a FEW such cross-code merges (and the reverse -- two
codes confirmed to have NO usable data despite looking like they should) were
genuinely checked, one at a time, rather than assumed.

Efficiency: both sources are far larger than any per-language word budget we need
(Morfessor saturates well before billions of words; low hundreds of thousands to a few
million is already a big improvement over what a small parallel corpus alone offers).
Every fetcher streams shard-by-shard and stops as soon as the word budget is hit, so a
high-resource language never downloads more than its first shard or two, instead of
the full multi-GB corpus -- discovery itself (listing which languages exist at all) is
cheap regardless of how many of the 636 a given run actually trains on.
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

# A handful of (source, source_language_code) merges/exclusions that needed a real,
# one-at-a-time judgment call rather than the general exact-code-match rule below --
# these WIN outright over whatever discover_morph_sources's own automatic merge would
# otherwise produce for the same code. Determined by directly querying both repos'
# file trees (not just card metadata, which was incomplete for Glot500) -- see
# conversation for the full derivation. MADLAD's generic "ar" is deliberately NOT used
# as a stand-in for arz (Egyptian Arabic): it's MSA-dominated, not the dialect, so
# using it would be a real variety mismatch, not just missing data -- the exact-match
# rule alone would also have MISSED merging eng/ben/spa/bam with MADLAD's own 2-letter
# "en"/"bn"/"es"/"bm" codes (different string, same language), which is why those are
# spelled out here explicitly instead of relying on the general rule for them.
_MANUAL_OVERRIDES = {
    "arz": [("glot500", "arz_Arab")],
    "bam": [("glot500", "bam_Latn"), ("madlad400", "bm")],
    "ben": [("madlad400", "bn")],
    "eng": [("glot500", "eng_Latn"), ("madlad400", "en")],
    "kas": [],  # no source has ANY text for this, confirmed live -- not a missing lookup
    "lij": [("glot500", "lij_Latn")],
    "mni": [("madlad400", "mni")],
    "nqo": [],  # no source has ANY text for this, confirmed live -- not a missing lookup
    "spa": [("glot500", "spa_Latn"), ("madlad400", "es")],
}


def list_madlad_langs():
    """Every language code allenai/MADLAD-400 offers, discovered from its own
    top-level "data-v1p5/" directory listing (non-recursive -- listing every FILE
    under data-v1p5/ takes minutes given ~280k files across the whole repo; listing
    just the per-language directories is fast, confirmed live) -- same "ask the
    source, don't hardcode a copy of it" convention common.data.corpora.
    list_glot500_configs already uses. Mixed ISO 639-1 (2-letter, e.g. "en"/"es"/"ar")
    and ISO 639-3 (3-letter, for languages with no 639-1 code, e.g. "abs"/"adh"/"ady")
    codes -- MADLAD doesn't use one convention uniformly."""
    tree = HfApi().list_repo_tree(MADLAD_REPO, path_in_repo="data-v1p5", repo_type="dataset")
    return sorted(item.path.rsplit("/", 1)[-1] for item in tree)


def discover_morph_sources(_cache={}):
    """Every language with usable monolingual text in EITHER Glot500 or MADLAD-400 --
    not just this project's original 9 hand-picked codes -- discovered live (see
    list_glot500_configs/list_madlad_langs above) and merged per the module docstring's
    own MERGE RULE (exact code-string match only, plus _MANUAL_OVERRIDES for the
    handful of genuinely-checked exceptions). Returns {lang_code: [(source_name,
    source_code), ...]}.

    One canonical script is picked per Glot500 short code when it offers more than
    one (confirmed live: 21 of 389 do) -- reuses common.data.oldi_data.LANG_SCRIPT's
    own already-vetted choice when the code is one of ITS codes (e.g. kas_Arab over
    kas_Deva), falling back to whichever variant sorts first alphabetically for every
    other code (an arbitrary but deterministic tiebreak, same spirit as
    common.data.corpora.list_smol_pairs's own mirror-pair dedup).

    Memoized (module-level _cache) since building this hits the network twice
    (Glot500's card_data, MADLAD's directory tree) -- cheap to call repeatedly within
    one process once discovered; call with an explicit `{}` first-arg override if you
    genuinely need to force a re-discovery (e.g. a long-running process across days).
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
    # character (common in Arabic-script/diacritic-heavy text especially) would
    # otherwise count as different word "types" for the exact same word.
    return WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def _iter_glot500_words(stem, max_words, _file_cache={}):
    # _file_cache={} is an intentional cross-call cache (the classic Python mutable-
    # default-argument gotcha, used deliberately here) -- without it, every one of the
    # ~7 languages we train would re-list Glot500's full ~1800-file tree from scratch.
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
    # Per-LANGUAGE directory listing (list_repo_tree, recursive under just
    # "data-v1p5/{code}"), not a repo-wide list_repo_files -- confirmed live
    # the whole repo has ~284k files across ~300 languages, so listing
    # everything just to filter down to one language's own files is a real,
    # avoidable cost at the scale discover_morph_sources now runs at
    # (hundreds of languages, not this project's original 9).
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
    """sources defaults to discover_morph_sources()[lang]; pass explicitly to override
    (e.g. for a language discovery doesn't find, or to force a specific choice).
    Returns a Counter, or an empty Counter if lang has no usable source (e.g. kas/nqo,
    see _MANUAL_OVERRIDES) -- callers check for emptiness, this never raises for a
    known-uncovered language."""
    sources = discover_morph_sources().get(lang, []) if sources is None else sources
    counts = Counter()
    for source_name, code in sources:
        counts.update(_FETCHERS[source_name](code, max_words_per_source))
    if max_word_types and len(counts) > max_word_types:
        # keep the most frequent types -- Morfessor's training cost scales with
        # lexicon size more than raw token count, so this bounds compute for
        # high-resource languages without touching the (already-fine) low-resource ones
        counts = Counter(dict(counts.most_common(max_word_types)))
    return counts


def train_morfessor(
    word_counts, freq_threshold=1, corpusweight=1.0, init_rand_split=0.5, seed=0
):
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
        freqthreshold=freq_threshold,
        init_rand_split=init_rand_split,
    )
    model.train_batch(algorithm="recursive")
    return model
