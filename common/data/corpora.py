"""Single registry of every text corpus this project draws on for TRAINING
(as opposed to common.data.oldi_data's own BOUQuET dev/test, a dedicated
held-out-eval concern, untouched here) -- used identically by tokenizer
training (common.data.cli_data.load_groups) and LLM pretraining
(systems.pretraining.data_prep). One shared interface for every source:

    stream_groups(source, langs=None, config=None, seed=0) -> Iterator[dict[str, str]]

Source categories:

  - PARALLEL_SOURCES = {oldi_seed, flores_dev}: a group has every requested
    language's own translation of the SAME content (see common.data.
    oldi_data for the join logic). These fully materialize to compute that
    join, so "streaming" here means iterating an already-loaded list, not
    lazy network streaming. Both default to `langs="all"`; common.data.
    oldi_data.LANGS (a fixed 9-code list, still used elsewhere) remains
    available as an explicit override.
  - MONOLINGUAL_SOURCES = {glot500, fineweb_edu, olmo_mix}: no cross-lingual
    alignment, one key per group. fineweb_edu/olmo_mix are lazily streamed
    from the HF Hub; multiple languages/configs are INTERLEAVED round-robin
    (not concatenated) so a --num-groups cap smaller than the per-language
    cap still sees a balanced mix rather than all of language A before
    language B. glot500 is the one exception: like bible_nlp/
    indigenous_panel below, it reads from a one-time LOCAL disk cache
    (common.data.prepare_glot500) instead of live streaming -- confirmed
    live that re-streaming its ~308GB/~411-config corpus from the HF Hub on
    every prep run (and on every RESUME's fast-forward) was the actual
    bottleneck of a real glot500-scale pretraining data prep. fineweb_edu/
    olmo_mix stay live-streamed (far larger, out of scope for that fix).
  - BITEXT_SOURCES = {smol, ccmatrix, un_pc, europarl, tatoeba_mt}: parallel
    like PARALLEL_SOURCES, but each group has exactly TWO keys (one
    language pair), lazily streamed like MONOLINGUAL_SOURCES -- `config`
    selects which pair (source-native naming, see list_bitext_configs/
    list_tatoeba_mt_pairs/list_smol_pairs); "all"/omitted round-robins
    every pair. smol lives here rather than in PARALLEL_SOURCES because
    intersecting IDs across its ~120 pairs simultaneously collapses the
    surviving intersection -- the same failure mode BOUQuET eval hit (fixed
    there by a union join, see _load_bouquet_split); per-pair streaming
    sidesteps it, at the cost of only 2 languages per group. ahelk/
    ccaligned_multilingual was considered and excluded: its HF repo only
    ships a deprecated loading script the installed `datasets` library
    refuses to run, and no adequate modern mirror was found.
  - STREAMED_PARALLEL_SOURCES = {bible_nlp}: like PARALLEL_SOURCES (one key
    per language, fully joined before yielding), but reads from a LOCAL
    disk cache built by a one-time common.data.prepare_bible_nlp run (which
    does the expensive part -- scanning the ~5.2GB source, picking one
    canonical translation per language -- once, offline). `langs` names an
    arbitrary subset of the ~1000+ available languages (no "all" default,
    even now that it's local: bible_nlp's translations don't share verse
    segmentation across languages, so intersecting too many risks the same
    "collapses to nothing" problem).
  - LOCAL_BITEXT_SOURCES = {indigenous_panel}: a small, curated panel of
    Indigenous (mostly polysynthetic) language pairs for a fairness
    comparison alongside BOUQuET (see common.data.indigenous_panel).
    Shaped like BITEXT_SOURCES, but like bible_nlp reads from a local disk
    cache (built by common.data.prepare_indigenous_panel) since each pair
    has its own bespoke access method with no live-streaming fallback.
    Unlike bible_nlp, no cross-language ID join is needed -- each pair's two
    languages are already row-aligned in their own source.

A single-language group runs fine through any trainer's plain next-byte CE
loss, but contributes nothing to a FAIRNESS loss term that compares
languages within a batch (fanta's Gini/anchor loss, flexitokens' boundary
hinge loss, fairtok's group-relative reward) -- those need genuinely
parallel content. This module draws that line by SOURCE, not by consumer:
nothing stops training fanta on fineweb_edu, it just won't help that loss
term.
"""

import itertools
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
LOCAL_BITEXT_SOURCES = {"indigenous_panel"}
ALL_SOURCES = sorted(
    PARALLEL_SOURCES
    | MONOLINGUAL_SOURCES
    | BITEXT_SOURCES
    | STREAMED_PARALLEL_SOURCES
    | LOCAL_BITEXT_SOURCES
    | {"synthetic"}
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
# together so a reader can see every external repo this module talks to in
# one place.
SMOL_REPO = "google/smol"
CCMATRIX_REPO = "sentence-transformers/parallel-sentences-ccmatrix"
UN_PC_REPO = "Helsinki-NLP/un_pc"
EUROPARL_REPO = "Helsinki-NLP/europarl"
TATOEBA_MT_REPO = "Helsinki-NLP/tatoeba_mt"
BIBLE_NLP_REPO = "bible-nlp/biblenlp-corpus"

# ccmatrix/un_pc/europarl configs live in card_data, discovered the same way
# as Glot500's; tatoeba_mt has no card_data configs (plain per-pair TSV
# files) so it gets its own file-listing-based lister below.
_CONFIG_BASED_BITEXT_REPOS = {"ccmatrix": CCMATRIX_REPO, "un_pc": UN_PC_REPO, "europarl": EUROPARL_REPO}


def list_glot500_configs():
    """All ~411 Glot500 language-script config names (e.g. "eng_Latn"),
    discovered live via the HF Hub API rather than a hardcoded snapshot."""
    info = HfApi().dataset_info("cis-lmu/Glot500")
    return sorted(c["config_name"] for c in info.card_data["configs"])


def _resolve_glot500_config(lang):
    """Accepts either a Glot500-native config name ("eng_Latn") or one of
    this project's own short codes ("eng") via common.data.oldi_data.
    LANG_SCRIPT; anything else passes through unchanged and fails at
    load_dataset time if it isn't a real Glot500 config."""
    return LANG_SCRIPT.get(lang, lang)


def list_bitext_configs(source):
    """Live pair-config names for ccmatrix/un_pc/europarl (e.g. "en-af",
    "ar-en", "bg-cs"), discovered via card_data."""
    if source not in _CONFIG_BASED_BITEXT_REPOS:
        raise ValueError(
            f"{source!r} has no card_data configs -- choose from {sorted(_CONFIG_BASED_BITEXT_REPOS)} "
            "(tatoeba_mt uses list_tatoeba_mt_pairs instead, see its own docstring)"
        )
    info = HfApi().dataset_info(_CONFIG_BASED_BITEXT_REPOS[source])
    return sorted(c["config_name"] for c in info.card_data["configs"])


def list_tatoeba_mt_pairs(split="test"):
    """Live file-listing discovery of available pairs for one tatoeba_mt
    split -- this repo has no card_data configs, just per-pair TSV files
    ("{split}/tatoeba-{split}.{pair}.tsv"), and no "train" split (only
    dev/test). Returns pairs in the dataset's own native naming (mostly ISO
    639-3, occasionally "_Script"-suffixed, a few non-ISO codes like "toki")
    -- opaque strings here, not validated against LANG_SCRIPT/LANGS."""
    if split not in ("dev", "test"):
        raise ValueError(f"tatoeba_mt has no {split!r} split -- choose 'dev' or 'test'")
    prefix, suffix = f"{split}/tatoeba-{split}.", ".tsv"
    files = HfApi().list_repo_files(TATOEBA_MT_REPO, repo_type="dataset")
    return sorted(f[len(prefix) : -len(suffix)] for f in files if f.startswith(prefix) and f.endswith(suffix))


def list_smol_pairs():
    """Live file-listing discovery of every pair google/smol's smolsent/
    directory offers (plain per-pair JSONL files, no card_data configs).

    Both directions of most pairs exist as separate files (byte-for-byte
    identical content with src/trg swapped, verified for 119 of 120 pairs;
    "en_tig" has no mirror) -- returning both would double-count each pair,
    so this dedupes to ONE canonical stem per unordered pair (whichever
    name sorts first alphabetically, an arbitrary but deterministic
    tiebreak). Not exclusively English-pivot despite the "smol" name --
    also includes Russian-pivot pairs and one Cantonese/Chinese pair with a
    different, non-parallel-sentence row schema (see _stream_smol_single,
    which skips rows it can't parse).
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
    deriving language codes from the filename (robust to whichever mirror
    direction list_smol_pairs kept). Silently skips rows missing "trg" (the
    one Cantonese/Chinese pair uses a different schema -- "trgs": a list, no
    "id" -- not worth special-casing for a single pair)."""
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
    dicts -- the shape MONOLINGUAL_SOURCES all reduce to."""
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
                # educational quality) -- "eng" reflects that, not a
                # per-row language detection.
                yield {"eng": row["text"]}
    elif source == "olmo_mix":
        config = config_or_lang or "default"
        if config not in OLMO_MIX_CONFIGS:
            raise ValueError(f"unknown olmo_mix config {config!r}; choose from {OLMO_MIX_CONFIGS}")
        for row in _stream_hf("allenai/olmo-mix-1124", config):
            if row.get("text"):
                # Predominantly English, but unlike fineweb_edu this is a
                # simplification, not a verified per-row claim -- OLMo-mix
                # has no reliable per-row language field to check.
                yield {"eng": row["text"]}
    else:
        raise ValueError(f"{source!r} is not a monolingual source")


def _stream_bitext_single(source, config):
    """One pair-config's own row stream for ccmatrix/un_pc/europarl,
    normalized to {lang: text} 2-key dicts. ccmatrix's own columns are
    always the generic "english"/"non_english" regardless of config (the
    non-English code comes from the config name, e.g. "en-af" -> "af");
    un_pc/europarl already yield a {translation: {code1: text1, code2:
    text2}} row, so `row["translation"]` IS the group, unchanged."""
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
    """tatoeba_mt has no card_data configs, so its `config` is a small
    "{split}/{pair-or-all}" DSL instead of a bare HF config name -- e.g.
    "test/deu-eng", "dev/all", or bare "all" (defaults to split "test", the
    larger split; tatoeba_mt has no "train" split so this creates no eval
    leakage)."""
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
    Trusts each row's own embedded language codes over re-deriving them
    from the filename (they always agree in practice, but the row is
    ground truth if they ever didn't)."""
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
# is usable (no live fallback -- see _load_bible_nlp_local).


def _load_bible_nlp_local(output_dir, langs):
    """Local disk N-way intersect-by-id join across `langs`, reading the
    per-language JSONL files common.data.prepare_bible_nlp already wrote
    (same {"id":..., "text":...} shape oldi_seed/flores_plus use) -- an
    ordinary local join now, not a live remote scan (the expensive
    full-file scan happens once in the prep script; see that module's
    docstring for the canonical-translation-per-language choice).

    Still requires an explicit, SMALL `langs` list (no "all" default):
    bible_nlp translations don't share verse segmentation across
    languages, so intersecting many at once risks the same "collapses to
    almost nothing" problem BOUQuET eval hit (see _load_bouquet_split) --
    training genuinely needs an N-way intersection, so the mitigation here
    is "ask for few languages," not a different join strategy.
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


GLOT500_LOCAL_DIR = "data/glot500"  # default local disk cache -- see
# common.data.prepare_glot500, which must be run once before this source is
# usable. No live fallback (see _stream_glot500_local_single): the live
# streaming path (list_glot500_configs/_stream_monolingual_single, still
# used below by fineweb_edu/olmo_mix and by prepare_glot500.py itself) was
# confirmed live to be the actual bottleneck of a real glot500-scale
# pretraining data prep -- ~308GB across ~411 lazy HF Hub streaming configs,
# re-fetched over the network from scratch on every prep run AND on every
# RESUME's fast-forward. A one-time local cache amortizes that cost across
# every future prep run instead of paying it every time.


def list_glot500_local_langs(output_dir=None):
    """Every language code that ACTUALLY has a locally prepared glot500
    file right now -- mirrors list_indigenous_panel_pairs: reads the
    directory listing, not list_glot500_configs()'s full ~411-language
    manifest, since prepare_glot500 may have been run with --langs/--limit
    for only some of them, or may still be mid-run (see that module's own
    resumability docstring)."""
    output_dir = output_dir or GLOT500_LOCAL_DIR
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        fname[: -len(".jsonl")] for fname in os.listdir(output_dir) if fname.endswith(".jsonl")
    )


def _stream_glot500_local_single(lang, output_dir=None):
    """Reads one language's already-{lang: text}-shaped JSONL file directly
    -- unlike bible_nlp, no cross-language join is needed (glot500 groups
    are single-key, one language per row)."""
    output_dir = output_dir or GLOT500_LOCAL_DIR
    path = os.path.join(output_dir, f"{lang}.jsonl")
    if not os.path.exists(path):
        raise ValueError(
            f"glot500 needs a one-time local prep step before it can be used -- {path!r} "
            f"doesn't exist. Run: python -m common.data.prepare_glot500 --output-dir {output_dir}"
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


INDIGENOUS_PANEL_LOCAL_DIR = "data/indigenous_panel"  # default local disk
# cache -- see common.data.prepare_indigenous_panel, which must be run once
# before this source is usable. No live fallback: every pair here has its
# own bespoke access method (see common.data.indigenous_panel.PAIRS), so
# there's no single live source to stream from directly.


def list_indigenous_panel_pairs(output_dir=None):
    """Every pair-code (e.g. "crk-en") that ACTUALLY has locally prepared
    data right now -- reads the directory listing, not PAIRS's full
    intended manifest, since prepare_indigenous_panel may have been run
    with --pairs for only some of them."""
    output_dir = output_dir or INDIGENOUS_PANEL_LOCAL_DIR
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        fname[: -len(".jsonl")] for fname in os.listdir(output_dir) if fname.endswith(".jsonl")
    )


def _stream_indigenous_panel_single(pair, output_dir=None):
    """Reads one pair's already-row-aligned {lang_a: text, lang_b: text}
    JSONL file directly -- unlike bible_nlp, no cross-language ID join is
    needed (each pair's two languages come pre-aligned from their own
    source)."""
    output_dir = output_dir or INDIGENOUS_PANEL_LOCAL_DIR
    path = os.path.join(output_dir, f"{pair}.jsonl")
    if not os.path.exists(path):
        raise ValueError(
            f"indigenous_panel needs a one-time local prep step before it can be used -- {path!r} "
            f"doesn't exist. Run: python -m common.data.prepare_indigenous_panel --output-dir {output_dir}"
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _round_robin(iterators):
    """Cycles through `iterators` one item at a time, dropping any that
    exhaust, until all are exhausted -- makes multi-language requests an
    interleaved, balanced mix rather than a sequential concatenation."""
    active = list(iterators)
    while active:
        for it in list(active):
            try:
                yield next(it)
            except StopIteration:
                active.remove(it)


def stream_groups(source, langs=None, config=None, seed=0, max_samples_per_pair=None):
    """The one entry point both common.data.cli_data.load_groups (tokenizer
    training) and systems.pretraining.data_prep (LLM pretraining) use.

    source: one of ALL_SOURCES.
    langs: language codes for synthetic/oldi_seed/flores_dev/glot500 --
    defaults to "all" for oldi_seed/flores_dev/glot500 when omitted;
    synthetic defaults to its own fake profile set. Also an arbitrary
    (small) language subset for bible_nlp (no "all" default -- see
    _stream_bible_nlp); `config` optionally overrides bible_nlp's OR
    glot500's local disk cache directory (default BIBLE_NLP_LOCAL_DIR /
    GLOT500_LOCAL_DIR respectively -- see _stream_glot500_local_single).
    Like bible_nlp, glot500 reads from a local disk cache (built by
    common.data.prepare_glot500) rather than live HF streaming -- no live
    fallback; see that module's own docstring for why. Ignored for
    fineweb_edu/olmo_mix and BITEXT_SOURCES, which select what they load
    via `config` instead: an HF config name for fineweb_edu/olmo_mix, a
    native pair name or "all"/omitted for smol/ccmatrix/un_pc/europarl, a
    "{split}/{pair-or-all}" string or "all"/omitted for tatoeba_mt, or a
    pair-code or "all"/omitted for indigenous_panel (same round-robin-
    over-pairs shape, reading from a local disk cache instead of live HF
    streaming).

    max_samples_per_pair: indigenous_panel ONLY (silently ignored by every
    other source) -- caps each pair's OWN row count to at most this many
    (first-N, not a random sample) before round-robining across pairs.
    Confirmed live to matter: arn-es (Mapudungun) alone has 256,992 rows,
    roughly as many as the rest of the 13-pair panel COMBINED -- for a
    rate-limited consumer (e.g. the Claude tokenizer's per-request API
    eval), this single pair dominates the round-robin's tail long past
    every other pair's exhaustion, making a full run impractically slow
    (the anchor-grouped evaluate_claude_on_indigenous_panel processes one
    whole anchor at a time, so this can also starve a smaller anchor of
    ever getting a turn -- see that function's own docstring). Irrelevant
    for free/fast local-compute consumers (HF-frontier, this project's own
    trained tokenizers), which already evaluate the panel at full scale
    without issue.

    Returns an iterator of {lang: text} dicts -- multi-key for
    PARALLEL_SOURCES/STREAMED_PARALLEL_SOURCES, 2-key for BITEXT_SOURCES/
    LOCAL_BITEXT_SOURCES, single-key for everything else.
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
        lang_list = list_glot500_local_langs(output_dir=config) if langs in (None, "all") else list(langs)
        if len(lang_list) == 1:
            yield from _stream_glot500_local_single(lang_list[0], output_dir=config)
            return
        yield from _round_robin(
            iter(_stream_glot500_local_single(lang, output_dir=config)) for lang in lang_list
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
    if source == "indigenous_panel":
        pairs = list_indigenous_panel_pairs() if config in (None, "all") else [config]

        def _pair_stream(pair):
            it = _stream_indigenous_panel_single(pair)
            return itertools.islice(it, max_samples_per_pair) if max_samples_per_pair else it

        if len(pairs) == 1:
            yield from _pair_stream(pairs[0])
            return
        yield from _round_robin(iter(_pair_stream(pair)) for pair in pairs)
        return
    raise ValueError(f"unknown source {source!r} -- choose from {ALL_SOURCES}")
