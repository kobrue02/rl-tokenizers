"""Downstream evaluation benchmark registry: XNLI, XCOPA, FLORES (MT), BLiMP,
CoLA, SQuAD -- the benchmarks in scope for evaluating a pretrained model
(systems.pretraining.train), as opposed to a tokenizer's own intrinsic
quality (common.eval.cross_tokenizer) or fitting one (systems/, data_prep.py).
Schemas confirmed against each source directly:

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
  - BLiMP: nyu-mll/blimp. 67 per-paradigm configs (linguistic phenomena,
    e.g. "adjunct_island") -- only ships a "train" split (a real quirk of
    the HF packaging: this is pure eval data, there's nothing to actually
    train on). Row: {sentence_good, sentence_bad, ...}. Scored as a
    2-choice MultipleChoiceExample with EMPTY shared context (see
    load_blimp) -- a minimal pair has no natural context/continuation
    split, so this compares each sentence's own full log-likelihood
    directly, the standard "simple LM method" BLiMP's own paper describes.
  - CoLA: nyu-mll/glue, config "cola". Row: {sentence, label}, label via
    ClassLabel(['unacceptable', 'acceptable']) (0/1). CONFIRMED LIVE:
    "test"'s labels are all -1 (GLUE's standard hidden-label leaderboard
    split) -- "validation" (1043 rows) is the real scored split, "train"
    (8551 rows) is used only for threshold calibration (see
    eval_harness.evaluate_cola).
  - SQuAD (v1.1): rajpurkar/squad. Row: {id, title, context, question,
    answers: {text: [...], answer_start: [...]}} -- multiple acceptable
    reference answer strings per question; official scoring (see
    eval_harness.evaluate_qa) takes the best match over all of them.

PROMPTING: XNLI/XCOPA's natural zero-shot templates use English scaffolding
words ("because", "Question:", "True, False, or Neither?"). Properly
localizing that per language would need verified translations this project
doesn't have. PROMPT_OVERRIDES lets a caller supply a per-language
template; languages without one fall back to the English template applied
to that language's own text -- linguistically imperfect but an honest
default, not a claim of faithful multilingual prompting. BLiMP/CoLA/SQuAD
are English-only benchmarks (general LM-quality checks, not part of this
project's own cross-lingual fairness comparison), so this doesn't apply to them.

CONTAMINATION: checked via systems.pretraining.cli_contamination -- an n-gram
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

# nyu-mll/blimp's 67 config names (linguistic paradigms), fetched live once
# and hardcoded here -- same convention as XNLI_LANGS/XCOPA_LANGS above,
# not re-discovered via HfApi on every call.
BLIMP_PARADIGMS = [
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction", "determiner_noun_agreement_1",
    "determiner_noun_agreement_2", "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2", "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause", "drop_argument", "ellipsis_n_bar_1",
    "ellipsis_n_bar_2", "existential_there_object_raising", "existential_there_quantifiers_1",
    "existential_there_quantifiers_2", "existential_there_subject_raising",
    "expletive_it_object_raising", "inchoative", "intransitive",
    "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question", "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2",
    "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
    "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
    "principle_A_reconstruction", "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2", "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope", "sentential_subject_island", "superlative_quantifiers_1",
    "superlative_quantifiers_2", "tough_vs_raising_1", "tough_vs_raising_2", "transitive",
    "wh_island", "wh_questions_object_gap", "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance",
]


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


@dataclasses.dataclass
class CoLAExample:
    """One linguistic-acceptability item: score `sentence`'s own unconditional
    log-likelihood (see eval_harness.evaluate_cola), compare to `label`
    (1=acceptable, 0=unacceptable, matching glue/cola's own ClassLabel order)."""

    lang: str
    sentence: str
    label: int


@dataclasses.dataclass
class QAExample:
    """One extractive-QA item: generate an answer to `question` given
    `context` (see eval_harness.evaluate_qa) and score against `answers`,
    every acceptable reference string for this question."""

    lang: str
    context: str
    question: str
    answers: list


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


def load_blimp(langs=None, split="train"):
    """langs: list of BLIMP_PARADIGMS names, defaults to all 67 (named `langs`,
    not `paradigms`, so this loader matches the same `langs=` kwarg every other
    _MULTIPLE_CHOICE_BENCHMARKS loader takes -- cli_eval's generic dispatch
    builds `kwargs = {"langs": ...}` for all of them). Yields
    MultipleChoiceExample with EMPTY context (a minimal pair has no natural
    shared context/continuation split -- this scores each of the two full
    sentences' own log-likelihood directly) and `.lang` repurposed as the
    PARADIGM name, not a real language code: BLiMP is English-only, but this
    lets evaluate_multiple_choice's existing per_language breakdown double
    as a per-paradigm accuracy breakdown for free, which is what BLiMP's own
    literature actually cares about (accuracy pooled across all 67 very
    different phenomena is a much less useful number). `label=0` always,
    since sentence_good is always the first choice."""

    def _one_paradigm(paradigm):
        ds = hf_datasets.load_dataset("nyu-mll/blimp", name=paradigm, split=split, streaming=True)
        for row in ds:
            yield MultipleChoiceExample(
                lang=paradigm,
                context="",
                choices=[row["sentence_good"], row["sentence_bad"]],
                label=0,
            )

    yield from _round_robin(_one_paradigm(p) for p in (langs or BLIMP_PARADIGMS))


def load_cola(split="validation"):
    """split: "validation" (default, 1043 rows) is the real scored split --
    glue/cola's own "test" split has every label set to -1 (GLUE's standard
    hidden-label leaderboard convention, confirmed live), so scoring against
    it would silently compare against a constant placeholder. "train" (8551
    rows) is used separately, for eval_harness.evaluate_cola's own threshold
    calibration, not for the reported number. Yields CoLAExample."""
    ds = hf_datasets.load_dataset("nyu-mll/glue", "cola", split=split, streaming=True)
    for row in ds:
        yield CoLAExample(lang="en", sentence=row["sentence"], label=row["label"])


def load_squad(split="validation"):
    """split: "validation" (the genuinely held-out split; SQuAD v1.1 has no
    "test" split with public labels at all). Yields QAExample, one per
    question -- rajpurkar/squad's own "answers" field already lists every
    acceptable reference answer string per question."""
    ds = hf_datasets.load_dataset("rajpurkar/squad", split=split, streaming=True)
    for row in ds:
        yield QAExample(
            lang="en", context=row["context"], question=row["question"],
            answers=list(row["answers"]["text"]),
        )


BENCHMARKS = {
    "xnli": load_xnli,
    "xcopa": load_xcopa,
    "flores_mt": load_flores_mt,
    "blimp": load_blimp,
    "cola": load_cola,
    "squad": load_squad,
}
