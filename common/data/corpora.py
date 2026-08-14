"""Single registry of every text corpus this project draws on for TRAINING
(as opposed to common.data.oldi_data's own BOUQuET dev/test, which stays a
dedicated held-out-evaluation concern, untouched here) -- used identically
by tokenizer training (common.data.cli_data.load_groups) and LLM pretraining
(pretraining.data_prep). One shared interface, not two separate per-purpose
registries: every source exposes the exact same shape,

    stream_groups(source, langs=None, config=None, seed=0) -> Iterator[dict[str, str]]

regardless of how differently each one is actually fetched underneath, and
regardless of which of the two consumers is asking. Two genuinely different
kinds of source exist, and this module is explicit about which is which
rather than papering over it:

  - PARALLEL_SOURCES = {oldi_seed, flores_dev, smol}: a group's dict has
    every requested language's own translation of the SAME content -- see
    common.data.oldi_data for the actual cross-lingual join logic. These already
    need to fully materialize to compute that join, so "streaming" here
    means "iterate an already-fully-loaded list", not lazy network
    streaming -- stated plainly, not hidden.
  - MONOLINGUAL_SOURCES = {glot500, fineweb_edu, olmo_mix}: no cross-lingual
    alignment exists to preserve, so every yielded group has exactly ONE
    key. Lazily streamed from the HF Hub (datasets.load_dataset(streaming=
    True)); when more than one language/config is requested, INTERLEAVED
    round-robin rather than concatenated, so a downstream consumer that
    only takes the first N groups still sees a balanced mix across
    languages rather than all of language A followed by all of language B
    -- confirmed this matters directly: an earlier version of this
    interleaving (for glot500 specifically) filled language 1 to its cap
    before touching language 2, and a --num-groups cap smaller than the
    per-language cap silently returned only the first language.
  - BITEXT_SOURCES = {ccmatrix, un_pc, europarl, tatoeba_mt}: genuinely
    parallel like PARALLEL_SOURCES, but each yielded group has exactly TWO
    keys (one specific language pair), lazily streamed like
    MONOLINGUAL_SOURCES rather than fully joined ahead of time -- `config`
    selects WHICH pair (the source's own native config/pair naming, not
    this project's own short codes -- see list_bitext_configs/
    list_tatoeba_mt_pairs), the same role `config` already plays for
    fineweb_edu/olmo_mix. `config="all"` round-robins every available pair
    for that source. ahelk/ccaligned_multilingual was investigated for
    this same list and deliberately excluded: its HF repo only ships a
    deprecated Python dataset-loading script, which the installed
    `datasets` library refuses to run at all (not a trust_remote_code
    gate -- confirmed live, it raises unconditionally), and no clean
    modern replacement or mirror of adequate/verified completeness was
    found (unlike tatoeba below).
  - STREAMED_PARALLEL_SOURCES = {bible_nlp}: like PARALLEL_SOURCES, every
    yielded group has one key per requested language (fully joined before
    yielding, not lazily streamed row-by-row) -- but sourced from a live,
    multi-gigabyte remote file rather than a small pre-packaged one, so
    `langs` names an arbitrary subset of the ~1000+ languages available
    (no built-in default panel -- see _stream_bible_nlp).

A single-language group runs fine through every systems/ trainer's plain
next-byte CE loss, but contributes nothing to any FAIRNESS loss term that
compares languages within one batch (fanta's Gini/anchor loss, flexitokens'
hinge loss, fairtok's group-relative reward) -- those need genuinely
parallel content. This module draws that line by SOURCE, not by consumer:
nothing here stops training fanta on fineweb_edu, it just won't do anything
useful for that specific loss term if you do -- the caller's job to pick a
source that fits what they're training, this module's job only to be
honest about which sources can supply what.
"""

import datasets as hf_datasets
from huggingface_hub import HfApi

from .synthetic import LANG_PROFILES, make_synthetic_parallel_groups
from .oldi_data import LANG_SCRIPT, LANGS, load_flores_plus, load_oldi_seed, load_smol_groups

PARALLEL_SOURCES = {"oldi_seed", "flores_dev", "smol"}
MONOLINGUAL_SOURCES = {"glot500", "fineweb_edu", "olmo_mix"}
BITEXT_SOURCES = {"ccmatrix", "un_pc", "europarl", "tatoeba_mt"}
STREAMED_PARALLEL_SOURCES = {"bible_nlp"}
ALL_SOURCES = sorted(
    PARALLEL_SOURCES | MONOLINGUAL_SOURCES | BITEXT_SOURCES | STREAMED_PARALLEL_SOURCES | {"synthetic"}
)

FINEWEB_EDU_CONFIGS = ["default", "sample-10BT", "sample-100BT", "sample-350BT"]
OLMO_MIX_CONFIGS = [
    "default",
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
]

# repo_id for each BITEXT_SOURCES/STREAMED_PARALLEL_SOURCES entry -- kept
# named/together so a reader can see every external repo this module talks
# to in one place, same convention as FINEWEB_EDU_CONFIGS/OLMO_MIX_CONFIGS
# above.
CCMATRIX_REPO = "sentence-transformers/parallel-sentences-ccmatrix"
UN_PC_REPO = "Helsinki-NLP/un_pc"
EUROPARL_REPO = "Helsinki-NLP/europarl"
TATOEBA_MT_REPO = "Helsinki-NLP/tatoeba_mt"
BIBLE_NLP_REPO = "bible-nlp/biblenlp-corpus"

# ccmatrix/un_pc/europarl configs live in card_data, discovered the same way
# list_glot500_configs discovers Glot500's; tatoeba_mt has no card_data
# configs (it ships plain per-pair TSV files, not HF-native configs) so it
# gets its own file-listing-based lister below instead.
_CONFIG_BASED_BITEXT_REPOS = {"ccmatrix": CCMATRIX_REPO, "un_pc": UN_PC_REPO, "europarl": EUROPARL_REPO}


def list_glot500_configs():
    """All ~411 Glot500 language-script config names (e.g. "eng_Latn"),
    discovered live via the HF Hub API -- no hardcoded snapshot, the same
    "ask the source, don't hardcode a copy of it" convention
    common.data.oldi_data._list_all_stems already uses for oldi_seed/flores_plus."""
    info = HfApi().dataset_info("cis-lmu/Glot500")
    return sorted(c["config_name"] for c in info.card_data["configs"])


def _resolve_glot500_config(lang):
    """Accepts either a Glot500-native config name directly ("eng_Latn") or,
    as a convenience, one of this project's own curated short codes ("eng")
    via common.data.oldi_data.LANG_SCRIPT -- anything else passes through
    unchanged and fails clearly at load_dataset time if it isn't a real
    Glot500 config."""
    return LANG_SCRIPT.get(lang, lang)


def list_bitext_configs(source):
    """Live pair-config names for ccmatrix/un_pc/europarl (e.g. "en-af",
    "ar-en", "bg-cs") -- discovered via card_data, confirmed present for all
    three repos, the same "ask the source, don't hardcode a copy of it"
    convention list_glot500_configs already uses."""
    if source not in _CONFIG_BASED_BITEXT_REPOS:
        raise ValueError(
            f"{source!r} has no card_data configs -- choose from {sorted(_CONFIG_BASED_BITEXT_REPOS)} "
            "(tatoeba_mt uses list_tatoeba_mt_pairs instead, see its own docstring)"
        )
    info = HfApi().dataset_info(_CONFIG_BASED_BITEXT_REPOS[source])
    return sorted(c["config_name"] for c in info.card_data["configs"])


def list_tatoeba_mt_pairs(split="test"):
    """Live file-listing discovery of available language pairs for one
    tatoeba_mt split -- confirmed live this repo has no card_data configs
    (it ships plain per-pair TSV files directly, one file per
    split/pair: "{split}/tatoeba-{split}.{pair}.tsv"), and no "train" split
    at all (only "dev"/"test" -- it's a held-out-sized comparison corpus,
    not a bulk training source). Returns pair strings in the dataset's own
    native naming (mostly ISO 639-3, occasionally with a "_Script" suffix
    like "ber_Latn" mirroring this project's own Glot500-style stems, and a
    handful of non-ISO codes like "toki" for Toki Pona -- confirmed live,
    not guessed -- so pairs are opaque strings here, not validated against
    LANG_SCRIPT/LANGS)."""
    if split not in ("dev", "test"):
        raise ValueError(f"tatoeba_mt has no {split!r} split -- choose 'dev' or 'test'")
    prefix, suffix = f"{split}/tatoeba-{split}.", ".tsv"
    files = HfApi().list_repo_files(TATOEBA_MT_REPO, repo_type="dataset")
    return sorted(f[len(prefix) : -len(suffix)] for f in files if f.startswith(prefix) and f.endswith(suffix))


def _stream_hf(repo_id, config, split="train"):
    return hf_datasets.load_dataset(repo_id, name=config, split=split, streaming=True)


def _stream_monolingual_single(source, config_or_lang):
    """One language/config's own row stream, normalized to {lang: text}
    dicts -- the per-source shape MONOLINGUAL_SOURCES all reduce to."""
    if source == "glot500":
        lang = config_or_lang
        for row in _stream_hf("cis-lmu/Glot500", _resolve_glot500_config(lang)):
            if row.get("text"):
                yield {lang: row["text"]}
    elif source == "fineweb_edu":
        config = config_or_lang or "sample-10BT"
        if config not in FINEWEB_EDU_CONFIGS:
            raise ValueError(f"unknown fineweb_edu config {config!r}; choose from {FINEWEB_EDU_CONFIGS}")
        for row in _stream_hf("HuggingFaceFW/fineweb-edu", config):
            if row.get("text"):
                # FineWeb-Edu is primarily English web text (filtered for
                # educational quality) -- "eng" here reflects that, not a
                # per-row language detection (the dataset DOES carry its
                # own "language" column, used for its own QA filtering, but
                # this module doesn't re-check it per row).
                yield {"eng": row["text"]}
    elif source == "olmo_mix":
        config = config_or_lang or "default"
        if config not in OLMO_MIX_CONFIGS:
            raise ValueError(f"unknown olmo_mix config {config!r}; choose from {OLMO_MIX_CONFIGS}")
        for row in _stream_hf("allenai/olmo-mix-1124", config):
            if row.get("text"):
                # Predominantly English, but unlike fineweb_edu this is a
                # SIMPLIFICATION, not a verified per-row claim -- OLMo-mix
                # has no reliable per-row language field across all 8
                # configs to check against.
                yield {"eng": row["text"]}
    else:
        raise ValueError(f"{source!r} is not a monolingual source")


def _stream_bitext_single(source, config):
    """One pair-config's own row stream for ccmatrix/un_pc/europarl,
    normalized to {lang: text} 2-key dicts -- confirmed live per source:
    ccmatrix's own columns are always the generic "english"/"non_english"
    regardless of config (the actual non-English code has to come from the
    config name itself, e.g. "en-af" -> "af"); un_pc/europarl both already
    yield a {translation: {code1: text1, code2: text2}} row keyed by the
    real pair codes, so `row["translation"]` IS the group, unchanged."""
    if source == "ccmatrix":
        _, other = config.split("-", 1)
        for row in _stream_hf(CCMATRIX_REPO, config):
            if row.get("english") and row.get("non_english"):
                yield {"en": row["english"], other: row["non_english"]}
    elif source in ("un_pc", "europarl"):
        repo = _CONFIG_BASED_BITEXT_REPOS[source]
        codes = config.split("-", 1)
        for row in _stream_hf(repo, config):
            translation = row.get("translation") or {}
            if all(translation.get(code) for code in codes):
                yield dict(translation)
    else:
        raise ValueError(f"{source!r} is not a config-based bitext source")


def _parse_tatoeba_mt_config(config):
    """tatoeba_mt has no card_data configs (see list_tatoeba_mt_pairs), so
    its own `config` string is a small "{split}/{pair-or-all}" DSL instead
    of a bare HF config name -- e.g. "test/deu-eng", "dev/all", or bare
    "all" (defaults to split "test", the larger of the two and the
    convention this project already uses elsewhere for held-out-style
    splits treated as a plain training source -- see flores_dev's own
    "dev is for training, devtest is reserved" split discipline; tatoeba_mt
    has no "train" split at all so this doesn't create any actual eval
    leakage, nothing else in this project reads tatoeba_mt as held-out)."""
    if config is None or config == "all":
        return "test", "all"
    if "/" not in config:
        raise ValueError(f"tatoeba_mt config must be '{{split}}/{{pair-or-all}}' or 'all', got {config!r}")
    split, pair = config.split("/", 1)
    if split not in ("dev", "test"):
        raise ValueError(f"tatoeba_mt has no {split!r} split -- choose 'dev' or 'test'")
    return split, pair


def _stream_tatoeba_mt_single(split, pair):
    """One pair's own TSV rows, normalized to {lang: text} 2-key dicts.
    Trusts each row's own embedded language codes (columns 1/2) over
    re-deriving them from the filename -- confirmed live these always
    agree, but the row is the ground truth if they ever didn't."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TATOEBA_MT_REPO, f"{split}/tatoeba-{split}.{pair}.tsv", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            src_lang, tgt_lang, src_text, tgt_text = parts
            if src_text and tgt_text:
                yield {src_lang: src_text, tgt_lang: tgt_text}


def _bible_nlp_canonical_entries(langs):
    """Streams bible-nlp/biblenlp-corpus's own top-level corpus.json (a
    live, ~5.2GB remote file -- confirmed via HfApi files_metadata) directly
    over HTTP via ijson, WITHOUT downloading it locally, extracting only
    the requested `langs`' own entries. Correct (not just an optimization)
    to stop as soon as every requested language's key has been seen: each
    top-level key is one distinct language appearing exactly once, so
    nothing later in the file could add a further match.

    Some languages have MULTIPLE distinct Bible translations mixed together
    under one key -- confirmed live, e.g. English alone spans 43 distinct
    "file" values across ~1.09M verse-entries. This picks whichever single
    translation has the MOST verse entries as that language's canonical
    text, the same kind of one-canonical-choice judgment call
    common.data.oldi_data.LANG_SCRIPT already makes for scripts.

    Memory note: this buffers one requested language's ENTIRE combined-
    across-translations entry list at a time (unavoidable -- ijson.kvitems
    yields each top-level key's value as one complete object), so requesting
    a very high-resource language like English costs real memory (~1.09M
    small dicts); requesting a handful of languages at a time is the
    intended use, not "give me every language at once".

    Cost note for a MISTYPED or genuinely absent language code: the early
    stop above only fires once every requested key has actually been seen,
    so a single wrong/absent code forces a full sequential scan of the
    entire ~5.2GB file before this raises in _stream_bible_nlp (confirmed
    live) -- there's no index to check membership any faster than that; get
    the language code right (or verify it's one of ~1000+ this repo
    actually has) before running a real job, not after.
    """
    import requests
    import ijson
    from huggingface_hub import hf_hub_url

    remaining = set(langs)
    canonical = {}
    url = hf_hub_url(BIBLE_NLP_REPO, "corpus.json", repo_type="dataset")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    try:
        for key, entries in ijson.kvitems(resp.raw, ""):
            if key not in remaining:
                continue
            by_file = {}
            for entry in entries:
                by_file.setdefault(entry["file"], []).append(entry)
            best_file = max(by_file, key=lambda f: len(by_file[f]))
            canonical[key] = {tuple(entry["verses"]): entry["text"] for entry in by_file[best_file]}
            remaining.discard(key)
            if not remaining:
                break
    finally:
        resp.close()
    return canonical


def _stream_bible_nlp(langs):
    """N-way parallel groups aligned by exact verse-key match across every
    requested language's canonical translation (see
    _bible_nlp_canonical_entries) -- a real, fully-in-memory join (like
    PARALLEL_SOURCES), not a lazy stream, since the join can't be computed
    without first collecting each language's full canonical entry set.
    verse-key is the entry's own "verses" tuple (usually one ref, sometimes
    a range) -- ranges that don't line up identically across two
    translations' own segmentation simply won't match here, a known,
    accepted limitation rather than a silent misalignment."""
    langs = list(langs)
    if len(langs) < 2:
        raise ValueError("bible_nlp needs at least 2 languages to form parallel groups")
    canonical = _bible_nlp_canonical_entries(langs)
    missing = [lang for lang in langs if lang not in canonical]
    if missing:
        raise ValueError(f"bible_nlp has no data for language(s) {missing}")
    verse_keys = set(canonical[langs[0]])
    for lang in langs[1:]:
        verse_keys &= set(canonical[lang])
    for verse_key in sorted(verse_keys):
        yield {lang: canonical[lang][verse_key] for lang in langs}


def _round_robin(iterators):
    """Cycles through `iterators` one item at a time, dropping any that
    exhaust, until all are exhausted -- what makes multi-language requests
    an interleaved, balanced mix rather than a sequential concatenation
    (see module docstring)."""
    active = list(iterators)
    while active:
        for it in list(active):
            try:
                yield next(it)
            except StopIteration:
                active.remove(it)


def stream_groups(source, langs=None, config=None, seed=0):
    """The one entry point both common.data.cli_data.load_groups (tokenizer
    training) and pretraining.data_prep (LLM pretraining) use.

    source: one of ALL_SOURCES.
    langs: language codes for synthetic/oldi_seed/flores_dev/smol/glot500
    (or "all" for glot500 -- every config list_glot500_configs finds), or
    an arbitrary language subset for bible_nlp (STREAMED_PARALLEL_SOURCES,
    no built-in default panel -- see _stream_bible_nlp). Ignored for
    fineweb_edu/olmo_mix and every BITEXT_SOURCES entry, which select what
    they load via `config` instead: an HF config name for fineweb_edu/
    olmo_mix (see FINEWEB_EDU_CONFIGS/OLMO_MIX_CONFIGS), a native pair name
    or "all" for ccmatrix/un_pc/europarl (see list_bitext_configs), or a
    "{split}/{pair-or-all}" string or bare "all" for tatoeba_mt (see
    _parse_tatoeba_mt_config/list_tatoeba_mt_pairs).

    Returns an iterator of {lang: text} dicts -- multi-key for
    PARALLEL_SOURCES/STREAMED_PARALLEL_SOURCES, 2-key for BITEXT_SOURCES,
    single-key for everything else (see module docstring).
    """
    if source == "synthetic":
        yield from make_synthetic_parallel_groups(
            400, langs=langs or list(LANG_PROFILES), seed=seed
        )
        return
    if source == "oldi_seed":
        yield from load_oldi_seed(langs=langs or LANGS)
        return
    if source == "flores_dev":
        yield from load_flores_plus(split="dev", langs=langs or LANGS)
        return
    if source == "smol":
        yield from load_smol_groups(langs=langs or [l for l in LANGS if l != "eng"])
        return
    if source == "glot500":
        lang_list = list_glot500_configs() if langs == "all" else list(langs or LANGS)
        if len(lang_list) == 1:
            yield from _stream_monolingual_single("glot500", lang_list[0])
            return
        yield from _round_robin(
            iter(_stream_monolingual_single("glot500", lang)) for lang in lang_list
        )
        return
    if source in ("fineweb_edu", "olmo_mix"):
        yield from _stream_monolingual_single(source, config)
        return
    if source in _CONFIG_BASED_BITEXT_REPOS:
        pairs = list_bitext_configs(source) if config in (None, "all") else [config]
        if len(pairs) == 1:
            yield from _stream_bitext_single(source, pairs[0])
            return
        yield from _round_robin(iter(_stream_bitext_single(source, pair)) for pair in pairs)
        return
    if source == "tatoeba_mt":
        split, pair_or_all = _parse_tatoeba_mt_config(config)
        pairs = list_tatoeba_mt_pairs(split) if pair_or_all == "all" else [pair_or_all]
        if len(pairs) == 1:
            yield from _stream_tatoeba_mt_single(split, pairs[0])
            return
        yield from _round_robin(iter(_stream_tatoeba_mt_single(split, pair)) for pair in pairs)
        return
    if source == "bible_nlp":
        if not langs:
            raise ValueError("bible_nlp requires --langs (an arbitrary subset -- no built-in default panel)")
        yield from _stream_bible_nlp(langs)
        return
    raise ValueError(f"unknown source {source!r} -- choose from {ALL_SOURCES}")
