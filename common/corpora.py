"""Single registry of every text corpus this project draws on for TRAINING
(as opposed to common.oldi_data's own BOUQuET dev/test, which stays a
dedicated held-out-evaluation concern, untouched here) -- used identically
by tokenizer training (common.cli_data.load_groups) and LLM pretraining
(pretraining.data_prep). One shared interface, not two separate per-purpose
registries: every source exposes the exact same shape,

    stream_groups(source, langs=None, config=None, seed=0) -> Iterator[dict[str, str]]

regardless of how differently each one is actually fetched underneath, and
regardless of which of the two consumers is asking. Two genuinely different
kinds of source exist, and this module is explicit about which is which
rather than papering over it:

  - PARALLEL_SOURCES = {oldi_seed, flores_dev, smol}: a group's dict has
    every requested language's own translation of the SAME content -- see
    common.oldi_data for the actual cross-lingual join logic. These already
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

from .data import LANG_PROFILES, make_synthetic_parallel_groups
from .oldi_data import LANG_SCRIPT, LANGS, load_flores_plus, load_oldi_seed, load_smol_groups

PARALLEL_SOURCES = {"oldi_seed", "flores_dev", "smol"}
MONOLINGUAL_SOURCES = {"glot500", "fineweb_edu", "olmo_mix"}
ALL_SOURCES = sorted(PARALLEL_SOURCES | MONOLINGUAL_SOURCES | {"synthetic"})

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


def list_glot500_configs():
    """All ~411 Glot500 language-script config names (e.g. "eng_Latn"),
    discovered live via the HF Hub API -- no hardcoded snapshot, the same
    "ask the source, don't hardcode a copy of it" convention
    common.oldi_data._list_all_stems already uses for oldi_seed/flores_plus."""
    info = HfApi().dataset_info("cis-lmu/Glot500")
    return sorted(c["config_name"] for c in info.card_data["configs"])


def _resolve_glot500_config(lang):
    """Accepts either a Glot500-native config name directly ("eng_Latn") or,
    as a convenience, one of this project's own curated short codes ("eng")
    via common.oldi_data.LANG_SCRIPT -- anything else passes through
    unchanged and fails clearly at load_dataset time if it isn't a real
    Glot500 config."""
    return LANG_SCRIPT.get(lang, lang)


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
    """The one entry point both common.cli_data.load_groups (tokenizer
    training) and pretraining.data_prep (LLM pretraining) use.

    source: one of ALL_SOURCES.
    langs: language codes for synthetic/oldi_seed/flores_dev/smol/glot500
    (or "all" for glot500 -- every config list_glot500_configs finds).
    Ignored for fineweb_edu/olmo_mix, which are single-language English
    sources selected by `config` instead (an HF config name -- see
    FINEWEB_EDU_CONFIGS/OLMO_MIX_CONFIGS).

    Returns an iterator of {lang: text} dicts -- multi-key for the
    PARALLEL_SOURCES, single-key for everything else (see module docstring).
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
    raise ValueError(f"unknown source {source!r} -- choose from {ALL_SOURCES}")
