"""Downstream evaluation benchmark registry: XNLI, XCOPA, FLORES (MT) -- the
three in scope for evaluating a pretrained model (pretraining.train), as
opposed to a tokenizer's own intrinsic quality (common.eval.cross_tokenizer)
or fitting one (systems/, data_prep.py). Schemas confirmed against each
source directly:

  - XNLI: facebook/xnli. 15 per-language configs (ar/bg/de/el/en/es/fr/hi/
    ru/sw/th/tr/ur/vi/zh), loaded individually rather than the differently-
    shaped pooled "all_languages" config, for a uniform row shape. Row:
    {premise, hypothesis, label}, label via ClassLabel(['entailment',
    'neutral', 'contradiction']).
  - XCOPA: cambridgeltl/xcopa. 11 real per-language configs (et/ht/id/it/
    qu/sw/ta/th/tr/vi/zh); "translation-X" MT variants not used. Row:
    {premise, choice1, choice2, question ('cause'/'effect'), label (0/1)}.
  - FLORES (MT): reuses common.data.oldi_data.load_flores_plus directly.
    load_flores_mt calls it with langs="all", which loads all ~227 native
    languages with zero id shrinkage -- every per-language file shares the
    same 997 sentence ids since FLORES is fully N-way parallel, so any
    language pair is valid. Accepts flores_plus's lang_Script stems (e.g.
    "deu_Latn") or LANG_SCRIPT's short codes (backward compat with oldi_
    seed/smol/BOUQuET tooling), auto-resolved. split="devtest" is the
    standard held-out MT split, disjoint from "dev" (what tokenizer
    training draws on).

PROMPTING: XNLI/XCOPA's natural zero-shot templates use English scaffolding
words ("because", "Question:", "True, False, or Neither?"). Properly
localizing that per language would need verified translations this project
doesn't have. PROMPT_OVERRIDES lets a caller supply a per-language
template; languages without one fall back to the English template applied
to that language's own text -- linguistically imperfect but an honest
default, not a claim of faithful multilingual prompting.

CONTAMINATION: checked via pretraining.cli_contamination -- an n-gram
overlap scan between any common.data.corpora source and these benchmarks'
examples (see contamination.py). FLORES-MT's "devtest" split is disjoint
from "dev" (tokenizer training's split), guarding one narrower leak beyond
the general scan. Run cli_contamination explicitly against whichever
source(s) fed a given pretraining run before trusting its eval numbers --
an unrun or --max-corpus-docs-capped scan tells you nothing either way.
"""

import dataclasses

import datasets as hf_datasets

from common.data.oldi_data import LANG_SCRIPT, load_flores_plus

XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
XCOPA_LANGS = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]

XNLI_LABEL_NAMES = ["entailment", "neutral", "contradiction"]  # ClassLabel
# order per facebook/xnli's declared features -- do not reorder.


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
    exhaust, until all are exhausted. Needed because a naively-concatenated
    multi-language stream under a global cap (--max-examples via
    itertools.islice) would silently draw only from the first language --
    confirmed on a real run (--langs en,de,fr,ar,zh --max-examples 1000
    came back with only "en")."""
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
    MultipleChoiceExample, per-language configs loaded one at a time,
    interleaved round-robin across languages (see _round_robin)."""

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


def _resolve_flores_lang(code):
    """Accepts either a short code LANG_SCRIPT maps (e.g. "eng" ->
    "eng_Latn", kept for backward compat with oldi_seed/smol/BOUQuET
    tooling) or a full lang_Script stem directly (e.g. "deu_Latn"). Passed
    through unchanged if not a known short code."""
    return LANG_SCRIPT.get(code, code)


def load_flores_mt(lang_pairs, split="devtest"):
    """lang_pairs: list of (source_lang, target_lang) tuples -- either
    short codes (see _resolve_flores_lang) or full flores_plus lang_Script
    stems (e.g. "eng_Latn"), any of its ~227 native languages.

    Loads the full 227-language set once (langs="all") regardless of how
    many pairs are requested, then slices out the requested pairs -- more
    expensive for a single pair but the only way to support arbitrary
    pairs (LANG_SCRIPT only maps 9 short codes). Per-language files are
    cached locally after first download. Pairs are interleaved round-robin
    (see _round_robin) so a global --max-examples cap samples every
    requested pair. Yields TranslationExample."""
    resolved_pairs = [(_resolve_flores_lang(s), _resolve_flores_lang(t)) for s, t in lang_pairs]
    groups = load_flores_plus(split=split, langs="all")
    available = set(groups[0]) if groups else set()
    for src, tgt in resolved_pairs:
        for code in (src, tgt):
            if code not in available:
                raise ValueError(
                    f"{code!r} is not one of flores_plus's own {len(available)} native "
                    f"language stems for split={split!r} (e.g. 'eng_Latn', 'deu_Latn') -- "
                    "see the openlanguagedata/flores_plus dataset's own file listing for valid stems"
                )

    def _one_pair(src, tgt):
        for group in groups:
            yield TranslationExample(
                source_lang=src, target_lang=tgt, source_text=group[src], reference_text=group[tgt]
            )

    yield from _round_robin(_one_pair(src, tgt) for src, tgt in resolved_pairs)


BENCHMARKS = {
    "xnli": load_xnli,
    "xcopa": load_xcopa,
    "flores_mt": load_flores_mt,
}
