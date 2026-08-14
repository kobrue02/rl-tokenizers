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

  - PARALLEL_SOURCES = {oldi_seed, flores_dev}: a group's dict has every
    requested language's own translation of the SAME content -- see
    common.data.oldi_data for the actual cross-lingual join logic. These already
    need to fully materialize to compute that join, so "streaming" here
    means "iterate an already-fully-loaded list", not lazy network
    streaming -- stated plainly, not hidden. Both default to `langs="all"`
    now (every language the source natively offers), not a curated panel --
    the old 9-language default (the strict cross-lingual intersection this
    project's own eval panel is built from) stays available as an explicit
    `--langs` override, it just isn't the default any more.
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
  - BITEXT_SOURCES = {smol, ccmatrix, un_pc, europarl, tatoeba_mt}: genuinely
    parallel like PARALLEL_SOURCES, but each yielded group has exactly TWO
    keys (one specific language pair), lazily streamed like
    MONOLINGUAL_SOURCES rather than fully joined ahead of time -- `config`
    selects WHICH pair (the source's own native config/pair naming, not
    this project's own short codes -- see list_bitext_configs/
    list_tatoeba_mt_pairs/list_smol_pairs), the same role `config` already
    plays for fineweb_edu/olmo_mix. `config="all"` (or omitted) round-robins
    every available pair for that source. smol lives here rather than in
    PARALLEL_SOURCES despite being genuinely N-way-joinable in principle
    (it's English-pivot bilingual pairs, structurally identical to
    ccmatrix) -- confirmed live that intersecting IDs across its ~120
    language pairs simultaneously (the old approach, on a curated 9-language
    panel) doesn't generalize to "every language it has": the more languages
    forced into one join, the smaller the surviving intersection gets,
    exactly the bug BOUQuET eval already hit and fixed by switching to a
    union join (see _load_bouquet_split's own docstring) -- per-pair
    streaming sidesteps this the same way ccmatrix/un_pc/europarl/
    tatoeba_mt already do, at the cost of each group only ever covering 2
    languages instead of however many a training run requests. ahelk/
    ccaligned_multilingual was investigated for this same list and
    deliberately excluded: its HF repo only ships a deprecated Python
    dataset-loading script, which the installed `datasets` library refuses
    to run at all (not a trust_remote_code gate -- confirmed live, it raises
    unconditionally), and no clean modern replacement or mirror of
    adequate/verified completeness was found (unlike tatoeba below).
  - STREAMED_PARALLEL_SOURCES = {bible_nlp}: like PARALLEL_SOURCES, every
    yielded group has one key per requested language (fully joined before
    yielding) -- but reads from a LOCAL disk cache that must be built first
    via a one-time run of common.data.prepare_bible_nlp (which does the
    genuinely expensive part -- scanning the ~5.2GB source file, picking one
    canonical translation per language -- ONCE, offline, not on every
    training run any more), rather than a small dataset this module can
    fetch directly the way oldi_seed/flores_plus can. `langs` names an
    arbitrary subset of the ~1000+ languages available (no "all" default,
    even now that it's local and fast to read -- see _load_bible_nlp_local
    for why: bible_nlp's translations don't share verse segmentation across
    languages by construction, so intersecting too many at once risks the
    same "collapses to almost nothing" problem BOUQuET eval hit before
    switching to a union join).

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

import json
import os

import datasets as hf_datasets
from huggingface_hub import HfApi

from .synthetic import LANG_PROFILES, make_synthetic_parallel_groups
from .oldi_data import LANG_SCRIPT, load_flores_plus, load_oldi_seed

PARALLEL_SOURCES = {"oldi_seed", "flores_dev"}
MONOLINGUAL_SOURCES = {"glot500", "fineweb_edu", "olmo_mix"}
BITEXT_SOURCES = {"smol", "ccmatrix", "un_pc", "europarl", "tatoeba_mt"}
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
SMOL_REPO = "google/smol"
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


def list_smol_pairs():
    """Live file-listing discovery of every language pair google/smol's own
    smolsent/ directory offers -- confirmed live this repo also has no
    card_data configs (plain per-pair JSONL files, "smolsent/{a}_{b}.jsonl"),
    the same reason tatoeba_mt gets its own file-listing lister rather than
    list_bitext_configs.

    Confirmed live: BOTH directions of most pairs exist as separate files
    (e.g. "af_en.jsonl" AND "en_af.jsonl" -- verified byte-for-byte identical
    content with src/trg swapped, not independently collected data, for
    119 of 120 pairs; "en_tig" is the one pair with no mirror). Returning
    both would silently double-count every shared pair as two "different"
    training sources of the exact same sentences, so this dedupes to ONE
    canonical stem per unordered pair (whichever of the two names sorts
    first alphabetically -- an arbitrary but deterministic and reproducible
    tiebreak, same spirit as LANG_SCRIPT's "one canonical script" choice).
    Not exclusively English-pivot despite the historical "smol" naming --
    also includes a handful of Russian-pivot pairs (e.g. "abq_ru") and a
    Cantonese/Chinese pair with its own different, non-parallel-sentence
    row schema (see _stream_smol_single, which skips rows it can't parse
    rather than guessing).
    """
    files = HfApi().list_repo_files(SMOL_REPO, repo_type="dataset")
    prefix, suffix = "smolsent/", ".jsonl"
    stems = [f[len(prefix) : -len(suffix)] for f in files if f.startswith(prefix) and f.endswith(suffix)]
    by_pair = {}
    for stem in stems:
        a, b = stem.split("_", 1)
        by_pair.setdefault(frozenset((a, b)), []).append(stem)
    return sorted(sorted(stems)[0] for stems in by_pair.values())


def _stream_smol_single(pair):
    """One pair's own JSONL rows, normalized to {lang: text} 2-key dicts --
    reads each row's own "sl"/"src"/"tl"/"trg" fields directly rather than
    deriving the two language codes from the filename (robust to either
    mirror direction being the one list_smol_pairs happened to keep).
    Silently skips rows missing "trg" (confirmed live: the one Cantonese/
    Chinese pair in this dataset uses a different schema entirely --
    "trgs": a list of references, no "id" -- rather than one sentence per
    row; not worth a special case for a single non-parallel-sentence pair)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SMOL_REPO, f"smolsent/{pair}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src, trg = row.get("src"), row.get("trg")
            if src and trg:
                yield {row["sl"]: src, row["tl"]: trg}


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


BIBLE_NLP_LOCAL_DIR = "data/bible_nlp"  # default local disk cache -- see
# common.data.prepare_bible_nlp, which must be run once before this source
# is usable at all (no live fallback -- see _load_bible_nlp_local).


def _load_bible_nlp_local(output_dir, langs):
    """Local disk N-way intersect-by-id join across `langs`, reading the
    per-language JSONL files common.data.prepare_bible_nlp already wrote --
    the exact same {"id": ..., "text": ...} per-line shape common.data.
    oldi_data._load_ngram_parallel's own oldi_seed/flores_plus files use, so
    this is an ordinary local join, not a live remote scan any more (that
    expensive full-file scan now happens ONCE, in the prep script, not on
    every training run -- see that module's own docstring for why, and for
    the canonical-translation-per-language choice already baked into these
    files).

    Still requires an explicit, SMALL `langs` list (no "all" default) even
    though the data is now local and fast to read: bible_nlp's various
    translations don't share verse segmentation across languages by
    construction the way oldi_seed/flores_plus do, so intersecting many
    languages at once risks the exact same "collapses to almost nothing"
    problem BOUQuET eval hit before switching to a union join (see
    common.data.oldi_data._load_bouquet_split's own docstring) -- this
    module doesn't attempt that fix here since, unlike BOUQuET eval, TRAINING
    genuinely needs an N-way intersection (see corpora.py's own module
    docstring on why PARALLEL_SOURCES groups must be aligned parallel
    content), so the real mitigation is "ask for few languages at a time",
    not a different join strategy.
    """
    if not os.path.isdir(output_dir):
        raise ValueError(
            f"bible_nlp needs a one-time local prep step before it can be used -- {output_dir!r} "
            f"doesn't exist. Run: python -m common.data.prepare_bible_nlp --output-dir {output_dir}"
        )
    per_lang = {}
    for lang in langs:
        path = os.path.join(output_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            raise ValueError(
                f"bible_nlp has no prepared data for language {lang!r} (looked for {path!r}) -- see "
                f"{os.path.join(output_dir, 'metadata.json')!r} for every language actually prepared"
            )
        with open(path, encoding="utf-8") as f:
            per_lang[lang] = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                per_lang[lang][row["id"]] = row["text"]
    common_ids = sorted(set.intersection(*(set(d) for d in per_lang.values())))
    return [{lang: per_lang[lang][i] for lang in langs} for i in common_ids]


def _stream_bible_nlp(langs, output_dir=None):
    langs = list(langs)
    if len(langs) < 2:
        raise ValueError("bible_nlp needs at least 2 languages to form parallel groups")
    yield from _load_bible_nlp_local(output_dir or BIBLE_NLP_LOCAL_DIR, langs)


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
    langs: language codes for synthetic/oldi_seed/flores_dev/glot500 --
    defaults to "all" (every language the source natively offers) for
    oldi_seed/flores_dev/glot500 when omitted; synthetic defaults to its own
    small fake profile set instead. Also an arbitrary (small -- see
    _load_bible_nlp_local) language subset for bible_nlp
    (STREAMED_PARALLEL_SOURCES, no "all" default even now that it's disk-
    cached -- see _stream_bible_nlp); `config` optionally overrides
    bible_nlp's own local disk cache directory (default BIBLE_NLP_LOCAL_DIR,
    "data/bible_nlp" -- see common.data.prepare_bible_nlp, which must be run
    once before this source is usable at all). Ignored for fineweb_edu/olmo_mix and every
    BITEXT_SOURCES entry, which select what they load via `config` instead:
    an HF config name for fineweb_edu/olmo_mix (see FINEWEB_EDU_CONFIGS/
    OLMO_MIX_CONFIGS), a native pair name or "all"/omitted for smol/
    ccmatrix/un_pc/europarl (see list_smol_pairs/list_bitext_configs), or a
    "{split}/{pair-or-all}" string or bare "all"/omitted for tatoeba_mt (see
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
        yield from load_oldi_seed(langs=langs or "all")
        return
    if source == "flores_dev":
        yield from load_flores_plus(split="dev", langs=langs or "all")
        return
    if source == "smol":
        pairs = list_smol_pairs() if config in (None, "all") else [config]
        if len(pairs) == 1:
            yield from _stream_smol_single(pairs[0])
            return
        yield from _round_robin(iter(_stream_smol_single(pair)) for pair in pairs)
        return
    if source == "glot500":
        lang_list = list_glot500_configs() if langs in (None, "all") else list(langs)
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
        yield from _stream_bible_nlp(langs, output_dir=config)
        return
    raise ValueError(f"unknown source {source!r} -- choose from {ALL_SOURCES}")
