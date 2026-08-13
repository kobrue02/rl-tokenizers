"""Downstream evaluation benchmark registry: XNLI, XCOPA, FLORES (MT) -- the
three named in scope for evaluating a PRETRAINED model (pretraining.train),
as opposed to everything else in this repo, which evaluates a TOKENIZER's
own intrinsic quality (common.eval_common) or fits one (systems/, this
package's own data_prep.py). Schemas confirmed directly against each source
before writing this (not assumed):

  - XNLI: facebook/xnli. 15 per-language configs (ar/bg/de/el/en/es/fr/hi/
    ru/sw/th/tr/ur/vi/zh) plus a differently-shaped "all_languages" pooled
    config (nested dicts, not flat rows) -- this module iterates the
    per-language configs individually rather than that pooled one, for a
    uniform row shape across languages. Row: {premise: str, hypothesis:
    str, label: int}, label via ClassLabel(['entailment', 'neutral',
    'contradiction']) -- confirmed via the dataset's own declared features,
    not assumed from the values alone.
  - XCOPA: cambridgeltl/xcopa. 11 real per-language configs (et/ht/id/it/
    qu/sw/ta/th/tr/vi/zh) plus "translation-X" machine-translated variants
    (not used here). Row: {premise, choice1, choice2, question ('cause' or
    'effect'), label (0 or 1, index of the correct choice)}.
  - FLORES (MT): reuses common.oldi_data.load_flores_plus directly rather
    than a separate loader -- flores_plus is already integrated in this
    project (systems/ tokenizer training also draws on it) and is
    genuinely N-way parallel. NOTE, checked directly rather than assumed:
    load_flores_plus's `langs` argument (when not "all") looks each code up
    in common.oldi_data.LANG_SCRIPT, which only has entries for this
    project's own established 9-language panel (LANGS/FLORES_MT_LANGS
    below) -- passing an arbitrary one of flores_plus's ~212 native
    lang_Script stems directly raises a KeyError, it does not silently
    work. So load_flores_mt here is restricted to that same 9-language
    panel (the one every other cross-lingual comparison in this project
    already uses via oldi_seed/smol/BOUQuET), not "any FLORES language" --
    broader coverage would need langs="all" plus client-side filtering,
    not implemented here since nothing else in this project needs it yet.
    split="devtest" (confirmed to exist as its own top-level split,
    distinct from "dev") is the standard FLORES held-out MT evaluation
    split -- "dev" is what systems/ tokenizer training itself already
    draws on, so devtest keeps this genuinely disjoint from anything the
    tokenizer or a pretraining run built from flores_plus's own "dev"
    split could have seen.

PROMPTING, a real judgment call stated rather than hidden: XNLI/XCOPA's
natural zero-shot templates (e.g. COPA's "{premise} because {choice}") use
English scaffolding words ("because", "so", "Question:", "True, False, or
Neither?"). Properly localizing that scaffolding per language would need
verified translations this project doesn't have for 11-15 languages: rather
than guess at them, PROMPT_OVERRIDES lets a caller supply a per-language
template, and every language without one falls back to the English
template applied to that language's own (non-English) premise/hypothesis/
choice text -- linguistically imperfect (English scaffolding around
non-English content), but honest about being a default, not a claim of
faithful multilingual prompting. Swap in real localized templates via
PROMPT_OVERRIDES before running a non-English XNLI/XCOPA eval that matters.

CONTAMINATION: checked, not just described, via pretraining.cli_contamination
-- an n-gram text-overlap scan between any common.corpora source (the SAME
sources a pretraining run actually trains on) and any of these three
benchmarks' own examples (see pretraining/contamination.py for the
detection approach: shingle the small benchmark side into an index, stream
the large corpus side once checking for matches). FLORES-MT specifically
also uses load_flores_plus's "devtest" split, disjoint from the "dev" split
systems/ tokenizer training draws on -- that guards against one additional,
narrower leak on top of the general n-gram scan. Run it explicitly (`python3
-m pretraining.cli_contamination --benchmark ... --corpus-dataset ...`)
against whichever source(s) actually fed a given pretraining run before
trusting that run's eval numbers -- a scan that was never run, or one
capped short of the full corpus via --max-corpus-docs, still tells you
nothing either way (see cli_contamination.py's own docstring for that
caveat).
"""

import dataclasses

import datasets as hf_datasets

from common.oldi_data import LANGS as FLORES_MT_LANGS
from common.oldi_data import load_flores_plus

XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
XCOPA_LANGS = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]

XNLI_LABEL_NAMES = ["entailment", "neutral", "contradiction"]  # ClassLabel order,
# confirmed via facebook/xnli's own declared features -- do not reorder.


@dataclasses.dataclass
class MultipleChoiceExample:
    """One zero-shot multiple-choice item: score every string in `choices`
    as a continuation of `context` (see eval_harness.loglikelihood), predict
    argmax, compare to `label` (an index into `choices`)."""

    lang: str
    context: str
    choices: list
    label: int


@dataclasses.dataclass
class TranslationExample:
    """One MT item: generate a translation of `source_text` (source_lang ->
    target_lang) and score against `reference_text` (see
    eval_harness.evaluate_translation)."""

    source_lang: str
    target_lang: str
    source_text: str
    reference_text: str


# {lang: (xnli_template, xcopa_cause_template, xcopa_effect_template)} --
# see module docstring. Empty by design: fill in real localized templates
# here as they become available, rather than shipping guessed ones.
PROMPT_OVERRIDES = {}


def _xnli_template(lang, premise, hypothesis):
    override = PROMPT_OVERRIDES.get(lang, {}).get("xnli")
    if override:
        return override(premise, hypothesis)
    return (
        f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\nAnswer:",
        [" True", " False", " Neither"],  # index-aligned with XNLI_LABEL_NAMES
    )


def _xcopa_template(lang, premise, choice1, choice2, question):
    override = PROMPT_OVERRIDES.get(lang, {}).get("xcopa")
    if override:
        return override(premise, choice1, choice2, question)
    connector = " because" if question == "cause" else " so"
    stem = premise.rstrip()
    if stem.endswith("."):
        stem = stem[:-1]
    stem += connector

    def _lowered(choice):
        return choice[0].lower() + choice[1:] if choice else choice

    return stem, [" " + _lowered(choice1), " " + _lowered(choice2)]


def _round_robin(iterables):
    """Cycles through `iterables` one item at a time, dropping any that
    exhaust, until all are exhausted -- the SAME fix common.corpora.
    _round_robin applies for Glot500 (see that module's docstring for the
    original incident): every loader below feeds one dataset/language PER
    ITEM lazily, in sequence, so a caller that applies a global cap (like
    pretraining.cli_eval's --max-examples, via itertools.islice) on a
    NAIVELY-CONCATENATED multi-language stream would silently only ever
    draw from the first language/pair before the cap is hit -- confirmed to
    actually happen this way in a real run (a --langs en,de,fr,ar,zh
    --max-examples 1000 XNLI eval came back with only "en" in
    per_language). Round-robining here, not by asking every caller to
    remember to cap per-language itself, is what makes any global
    --max-examples value actually sample every requested language/pair."""
    iterators = [iter(it) for it in iterables]
    active = list(iterators)
    while active:
        for it in list(active):
            try:
                yield next(it)
            except StopIteration:
                active.remove(it)


def load_xnli(langs=None, split="test"):
    """langs: list of XNLI_LANGS codes, defaults to all 15. Yields
    MultipleChoiceExample, per-language configs loaded one at a time (not
    the differently-shaped "all_languages" pooled config -- see module
    docstring), interleaved round-robin across languages (see
    _round_robin's own docstring for why that matters under a global cap)."""

    def _one_lang(lang):
        ds = hf_datasets.load_dataset("facebook/xnli", name=lang, split=split, streaming=True)
        for row in ds:
            context, choices = _xnli_template(lang, row["premise"], row["hypothesis"])
            yield MultipleChoiceExample(lang=lang, context=context, choices=choices, label=row["label"])

    yield from _round_robin(_one_lang(lang) for lang in (langs or XNLI_LANGS))


def load_xcopa(langs=None, split="test"):
    """langs: list of XCOPA_LANGS codes, defaults to all 11. Yields
    MultipleChoiceExample, interleaved round-robin across languages (see
    _round_robin)."""

    def _one_lang(lang):
        ds = hf_datasets.load_dataset("cambridgeltl/xcopa", name=lang, split=split, streaming=True)
        for row in ds:
            context, choices = _xcopa_template(
                lang, row["premise"], row["choice1"], row["choice2"], row["question"]
            )
            yield MultipleChoiceExample(lang=lang, context=context, choices=choices, label=row["label"])

    yield from _round_robin(_one_lang(lang) for lang in (langs or XCOPA_LANGS))


def load_flores_mt(lang_pairs, split="devtest"):
    """lang_pairs: list of (source_lang, target_lang) short-code tuples,
    each code one of FLORES_MT_LANGS (this project's established 9-language
    panel -- see module docstring for why arbitrary FLORES language codes
    aren't supported here). Loads each PAIR independently (a fresh
    load_flores_plus call per pair, not one load for the union of every
    language involved) so this stays correct and simple even when different
    pairs share a language; flores_plus's own per-language files are cached
    locally after the first download (see common.oldi_data._download), so
    repeated pairs sharing a language don't re-download it. Pairs are
    interleaved round-robin (see _round_robin) for the same reason
    load_xnli/load_xcopa are -- a global --max-examples cap should sample
    every requested pair, not just the first. Yields TranslationExample."""
    for src, tgt in lang_pairs:
        for code in (src, tgt):
            if code not in FLORES_MT_LANGS:
                raise ValueError(
                    f"{code!r} is not in this project's FLORES language panel "
                    f"{FLORES_MT_LANGS} -- see module docstring"
                )

    def _one_pair(src, tgt):
        groups = load_flores_plus(split=split, langs=[src, tgt])
        for group in groups:
            yield TranslationExample(
                source_lang=src, target_lang=tgt, source_text=group[src], reference_text=group[tgt]
            )

    yield from _round_robin(_one_pair(src, tgt) for src, tgt in lang_pairs)


BENCHMARKS = {
    "xnli": load_xnli,
    "xcopa": load_xcopa,
    "flores_mt": load_flores_mt,
}
