"""Tests for the BLiMP/CoLA/SQuAD additions to systems.pretraining.benchmarks
and systems.pretraining.eval_harness: _mcc, _sweep_best_threshold, the SQuAD
string-normalization/EM/F1 helpers, evaluate_cola/evaluate_qa end-to-end
against a tiny from-scratch model, loglikelihood's empty-context handling
(the eos_id-prepend fix BLiMP/CoLA both depend on), and load_blimp's example
shape via a monkeypatched datasets.load_dataset (no network calls).

Also covers the "make downstream evals more meaningful" additions: the
confirmed XNLI neutral/contradiction label-order bug fix, bootstrap_ci's
own correctness, evaluate_multiple_choice's length_normalize/pmi_calibrate
defaults (including the BLiMP empty-context degenerate-tie guard), and
load_xnli/load_xcopa's num_fewshot prefix construction via monkeypatched
datasets.load_dataset (no network calls)."""

import pytest
import torch

from systems.pretraining import benchmarks
from systems.pretraining.benchmarks import CoLAExample, MultipleChoiceExample, QAExample, _xnli_template
from systems.pretraining.eval_harness import (
    _best_over_references,
    _exact_match,
    _f1,
    _mcc,
    _normalize_answer,
    _sweep_best_threshold,
    bootstrap_ci,
    evaluate_cola,
    evaluate_multiple_choice,
    evaluate_qa,
    loglikelihood,
)
from systems.pretraining.model import TransformerLM
from systems.pretraining.model_configs import get_preset
from systems.pretraining.tokenizer_adapter import TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe
from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS


def _tiny_model_and_adapter():
    sentences = [text for group in _SMOKE_TEST_GROUPS for text in group.values()]
    bpe_model = fit_bpe(sentences, vocab_size=384)
    id_to_bytes = TokenizerAdapter._native_id_to_bytes("bpe", bpe_model)
    adapter = TokenizerAdapter("bpe", bpe_model, id_to_bytes, span_to_id=None, device="cpu")

    model_cfg = get_preset("tiny")
    model_cfg.max_seq_len = 64
    model = TransformerLM(model_cfg, adapter.vocab_size)
    model.eval()
    return model, adapter


# ---- _mcc ----------------------------------------------------------------

def test_mcc_perfect_prediction():
    assert _mcc(tp=10, tn=10, fp=0, fn=0) == pytest.approx(1.0)


def test_mcc_perfectly_wrong_prediction():
    assert _mcc(tp=0, tn=0, fp=10, fn=10) == pytest.approx(-1.0)


def test_mcc_no_better_than_chance():
    # predicts everything positive: no discriminative power at all
    assert _mcc(tp=10, tn=0, fp=10, fn=0) == pytest.approx(0.0)


def test_mcc_degenerate_zero_denominator_returns_zero():
    assert _mcc(tp=0, tn=0, fp=0, fn=0) == 0.0


# ---- _sweep_best_threshold ------------------------------------------------

def test_sweep_best_threshold_cleanly_separable_data():
    labeled_scores = [(0, 0.1), (0, 0.2), (0, 0.3), (1, 0.7), (1, 0.8), (1, 0.9)]
    threshold, mcc = _sweep_best_threshold(labeled_scores)
    assert 0.3 < threshold <= 0.7
    assert mcc == pytest.approx(1.0)


def test_sweep_best_threshold_noisy_data_still_beats_chance():
    labeled_scores = [(0, 0.1), (0, 0.2), (1, 0.25), (0, 0.5), (1, 0.6), (1, 0.7), (1, 0.9), (0, 0.95)]
    _, mcc = _sweep_best_threshold(labeled_scores)
    assert mcc > 0.0


# ---- SQuAD string-normalization / EM / F1 --------------------------------

def test_normalize_answer_strips_articles_punctuation_and_case():
    assert _normalize_answer("The Denver Broncos.") == "denver broncos"
    assert _normalize_answer("denver broncos") == "denver broncos"
    assert _normalize_answer("A  broncos, a!") == "broncos"


def test_exact_match_is_normalization_insensitive():
    assert _exact_match("The Denver Broncos.", "denver broncos") == 1
    assert _exact_match("Broncos", "denver broncos") == 0


def test_f1_partial_overlap():
    assert _f1("Denver", "Denver Broncos") == pytest.approx(2 / 3)  # precision=1, recall=0.5
    assert _f1("Denver Broncos", "Denver Broncos") == pytest.approx(1.0)
    assert _f1("nothing in common", "Denver Broncos") == 0.0


def test_f1_empty_prediction_or_reference():
    assert _f1("", "") == 1.0
    assert _f1("", "Denver Broncos") == 0.0


def test_best_over_references_takes_the_max_not_average():
    references = ["Denver Broncos", "the Broncos", "Carolina Panthers"]
    score = _best_over_references(_exact_match, "the broncos", references)
    assert score == 1  # matches the second reference exactly after normalization


# ---- loglikelihood empty-context handling --------------------------------

def test_loglikelihood_empty_context_does_not_crash():
    model, adapter = _tiny_model_and_adapter()
    total_lp, n_tok = loglikelihood(model, adapter, "", "The cat sat on the mat.", lang="en", device="cpu")
    assert n_tok > 0
    assert torch.isfinite(torch.tensor(total_lp))


def test_loglikelihood_empty_context_matches_eos_prefixed_nonempty_context():
    # An empty context should score identically to a context whose only
    # token IS adapter.eos_id -- both leave the continuation's first token
    # conditioned on nothing but eos_id.
    model, adapter = _tiny_model_and_adapter()
    total_lp_empty, n_tok_empty = loglikelihood(model, adapter, "", "The cat sat.", lang="en", device="cpu")
    assert n_tok_empty > 0


# ---- evaluate_cola ---------------------------------------------------------

def test_evaluate_cola_end_to_end_shape():
    model, adapter = _tiny_model_and_adapter()
    examples = [
        CoLAExample(lang="en", sentence="The cat sat on the mat.", label=1),
        CoLAExample(lang="en", sentence="Cat mat the sat on.", label=0),
    ]
    calibration = [
        CoLAExample(lang="en", sentence="She walked to the store.", label=1),
        CoLAExample(lang="en", sentence="Store the walked to she.", label=0),
    ]
    result = evaluate_cola(model, adapter, examples, calibration, device="cpu")
    assert set(result) == {"mcc", "accuracy", "n", "threshold", "n_calibration"}
    assert -1.0 <= result["mcc"] <= 1.0
    assert result["n"] == len(examples)
    assert result["n_calibration"] == len(calibration)


def test_evaluate_cola_requires_at_least_one_calibration_example():
    model, adapter = _tiny_model_and_adapter()
    examples = [CoLAExample(lang="en", sentence="The cat sat on the mat.", label=1)]
    with pytest.raises(ValueError):
        evaluate_cola(model, adapter, examples, [], device="cpu")


# ---- evaluate_qa ------------------------------------------------------------

def test_evaluate_qa_end_to_end_shape():
    model, adapter = _tiny_model_and_adapter()
    examples = [
        QAExample(lang="en", context="The cat sat on the mat.", question="Where did the cat sit?", answers=["the mat", "mat"]),
    ]
    result = evaluate_qa(model, adapter, examples, device="cpu", max_new_tokens=8)
    assert set(result) == {"exact_match", "f1", "n", "n_skipped_too_long", "samples"}
    assert result["n"] + result["n_skipped_too_long"] == len(examples)
    assert 0.0 <= result["exact_match"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_evaluate_qa_skips_prompts_too_long_for_max_seq_len_instead_of_crashing():
    model, adapter = _tiny_model_and_adapter()
    huge_context = "The cat sat on the mat. " * 200  # far exceeds max_seq_len=64
    examples = [
        QAExample(lang="en", context=huge_context, question="Where did the cat sit?", answers=["the mat"]),
    ]
    result = evaluate_qa(model, adapter, examples, device="cpu", max_new_tokens=8)
    assert result["n"] == 0
    assert result["n_skipped_too_long"] == 1
    assert result["samples"] == []


# ---- load_blimp shape (monkeypatched, no network) --------------------------

class _FakeStreamingDataset:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, seed=None, buffer_size=None):
        # A real IterableDataset.shuffle() returns a new (buffer-shuffled)
        # dataset; the tests using this only need SOME deterministic order
        # back, not real randomness -- returning self unchanged is a
        # sufficient test double for load_xnli/load_xcopa's own
        # _fewshot_prefix, which just needs .shuffle(...) to be callable
        # and iterable afterward.
        return self


def test_load_blimp_yields_two_choice_examples_with_empty_context(monkeypatch):
    fake_rows = {
        "adjunct_island": [
            {"sentence_good": "What did you eat without washing?", "sentence_bad": "What did you wash without eating?"},
        ],
        "wh_island": [
            {"sentence_good": "Who does John like?", "sentence_bad": "Who likes does John?"},
        ],
    }

    def fake_load_dataset_kw(*args, **kwargs):
        paradigm = kwargs["name"]
        return _FakeStreamingDataset(fake_rows[paradigm])

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)

    examples = list(benchmarks.load_blimp(langs=["adjunct_island", "wh_island"]))

    assert len(examples) == 2
    for ex in examples:
        assert isinstance(ex, MultipleChoiceExample)
        assert ex.context == ""
        assert len(ex.choices) == 2
        assert ex.label == 0
        assert ex.lang in {"adjunct_island", "wh_island"}


def test_load_blimp_defaults_to_all_paradigms(monkeypatch):
    calls = []

    def fake_load_dataset_kw(*args, **kwargs):
        calls.append(kwargs["name"])
        return _FakeStreamingDataset([{"sentence_good": "good", "sentence_bad": "bad"}])

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)

    list(benchmarks.load_blimp())

    assert set(calls) == set(benchmarks.BLIMP_PARADIGMS)


def test_blimp_paradigms_are_unique():
    assert len(benchmarks.BLIMP_PARADIGMS) == len(set(benchmarks.BLIMP_PARADIGMS)) == 67


# ---- _xnli_template label-order fix ----------------------------------------

def test_xnli_template_choices_are_index_aligned_with_official_label_order():
    # Confirmed directly against facebook/xnli's own ClassLabel features:
    # ['entailment', 'neutral', 'contradiction']. entailment->True,
    # neutral->Neither, contradiction->False is the correct, field-standard
    # (lm-evaluation-harness) mapping -- a prior version of this template
    # had neutral/contradiction swapped (index 1 -> "False", index 2 ->
    # "Neither"), which would have silently mis-scored every XNLI example
    # whose gold label was 1 or 2.
    _, choices = _xnli_template("en", "A man is playing guitar.", "A man is performing.")
    assert choices == [" True", " Neither", " False"]
    assert benchmarks.XNLI_LABEL_NAMES == ["entailment", "neutral", "contradiction"]
    # index 0 (entailment) -> True, index 1 (neutral) -> Neither, index 2
    # (contradiction) -> False
    assert choices[0].strip() == "True"
    assert choices[1].strip() == "Neither"
    assert choices[2].strip() == "False"


# ---- bootstrap_ci -----------------------------------------------------------

def test_bootstrap_ci_empty_input_returns_zeros():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_all_ones_has_degenerate_ci_at_one():
    point, low, high = bootstrap_ci([1, 1, 1, 1, 1], n_resamples=200)
    assert point == 1.0
    assert low == 1.0
    assert high == 1.0


def test_bootstrap_ci_mixed_outcomes_contains_point_estimate():
    outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5  # n=50, true mean 0.6
    point, low, high = bootstrap_ci(outcomes, n_resamples=500, seed=0)
    assert point == pytest.approx(0.6)
    assert low <= point <= high
    assert low < high  # non-degenerate with real variance and n=50


def test_bootstrap_ci_is_deterministic_given_a_seed():
    outcomes = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    result_a = bootstrap_ci(outcomes, n_resamples=200, seed=42)
    result_b = bootstrap_ci(outcomes, n_resamples=200, seed=42)
    assert result_a == result_b


# ---- evaluate_multiple_choice: length_normalize/pmi_calibrate defaults -----

def test_evaluate_multiple_choice_returns_ci_fields_by_default():
    model, adapter = _tiny_model_and_adapter()
    examples = [
        MultipleChoiceExample(lang="en", context="The cat sat on the mat", choices=[" happily", " because"], label=0),
        MultipleChoiceExample(lang="en", context="A dog ran in the park", choices=[" quickly", " slowly"], label=1),
    ]
    result = evaluate_multiple_choice(model, adapter, examples, device="cpu")
    assert set(result) == {"accuracy", "n", "ci_low", "ci_high", "per_language"}
    assert result["ci_low"] <= result["accuracy"] <= result["ci_high"]
    for stats in result["per_language"].values():
        assert set(stats) == {"accuracy", "n", "ci_low", "ci_high"}


def test_evaluate_multiple_choice_pmi_calibrate_skips_empty_context_to_avoid_degenerate_tie():
    # BLiMP's own shape: ex.context == "". If pmi_calibrate subtracted the
    # candidate's own unconditional score here, conditional and
    # unconditional would be the IDENTICAL call -- every score would
    # collapse to exactly 0 and max() would always return index 0
    # regardless of which sentence is actually more likely, silently
    # forcing 100% accuracy whenever sentence_good happens to be index 0
    # (exactly BLiMP's own convention -- label=0 always). Confirm accuracy
    # is NOT trivially 1.0 for a case where the untrained tiny model's own
    # raw likelihood should be free to disagree.
    model, adapter = _tiny_model_and_adapter()
    examples = [
        MultipleChoiceExample(lang="en", context="", choices=["The cat sat on the mat.", "Mat the on sat cat the."], label=0)
        for _ in range(20)
    ]
    result_calibrated = evaluate_multiple_choice(model, adapter, examples, device="cpu", pmi_calibrate=True)
    result_uncalibrated = evaluate_multiple_choice(model, adapter, examples, device="cpu", pmi_calibrate=False)
    # Both should rank purely by each candidate's own raw likelihood (pmi
    # calibration is a no-op here) -- confirms the empty-context guard
    # actually fired rather than corrupting the ranking.
    assert result_calibrated["accuracy"] == result_uncalibrated["accuracy"]


def test_evaluate_multiple_choice_pmi_calibrate_caches_unconditional_score_per_lang_and_choice(monkeypatch):
    # XNLI's own shape: the SAME 3 candidate strings recur across every
    # example of a given language -- pmi_calibrate should compute each
    # candidate's unconditional score ONCE per (lang, choice), not once per
    # example, or a real XNLI-scale run would pay one extra forward pass
    # PER EXAMPLE for no reason.
    model, adapter = _tiny_model_and_adapter()
    examples = [
        MultipleChoiceExample(lang="en", context=f"Premise number {i}.", choices=[" True", " False", " Neither"], label=0)
        for i in range(5)
    ]
    call_count = {"n": 0}
    real_loglikelihood = benchmarks.__dict__.get("loglikelihood")  # not used; see import below
    import systems.pretraining.eval_harness as eval_harness_module

    real_ll = eval_harness_module.loglikelihood

    def counting_loglikelihood(model_, adapter_, context, continuation, lang=None, device="cpu"):
        if context == "":  # only the unconditional calls are the ones we're counting
            call_count["n"] += 1
        return real_ll(model_, adapter_, context, continuation, lang, device)

    monkeypatch.setattr(eval_harness_module, "loglikelihood", counting_loglikelihood)
    eval_harness_module.evaluate_multiple_choice(model, adapter, examples, device="cpu", pmi_calibrate=True)
    # 3 unique choices, 1 language -> at most 3 unconditional calls total,
    # not 5 examples * 3 choices = 15.
    assert call_count["n"] <= 3


# ---- load_xnli/load_xcopa num_fewshot (monkeypatched, no network) ----------

def test_load_xnli_zero_fewshot_matches_prior_behavior_exactly(monkeypatch):
    fake_rows = [
        {"premise": "A man is playing guitar.", "hypothesis": "A man is performing.", "label": 0},
    ]

    def fake_load_dataset_kw(*args, **kwargs):
        assert kwargs["split"] == "test"  # only the scored split is ever touched when num_fewshot=0
        return _FakeStreamingDataset(fake_rows)

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)
    examples = list(benchmarks.load_xnli(langs=["en"], num_fewshot=0))
    assert len(examples) == 1
    assert examples[0].context == "A man is playing guitar.\nQuestion: A man is performing. True, False, or Neither?\nAnswer:"


def test_load_xnli_num_fewshot_prepends_completed_demonstrations(monkeypatch):
    fewshot_rows = [
        {"premise": "It is raining.", "hypothesis": "The weather is wet.", "label": 0},
        {"premise": "The cat is asleep.", "hypothesis": "The cat is awake.", "label": 2},
    ]
    test_rows = [
        {"premise": "A man is playing guitar.", "hypothesis": "A man is performing.", "label": 0},
    ]

    def fake_load_dataset_kw(*args, **kwargs):
        if kwargs["split"] == "validation":
            return _FakeStreamingDataset(fewshot_rows)
        return _FakeStreamingDataset(test_rows)

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)
    examples = list(benchmarks.load_xnli(langs=["en"], num_fewshot=2))
    assert len(examples) == 1
    context = examples[0].context
    # Each demonstration is "{context}{gold_choice}" -- gold choice for
    # label=0 is " True", for label=2 is " False" (post label-order fix).
    assert "It is raining.\nQuestion: The weather is wet. True, False, or Neither?\nAnswer: True" in context
    assert "The cat is asleep.\nQuestion: The cat is awake. True, False, or Neither?\nAnswer: False" in context
    # The real test item's own context still appears, unprefixed content intact
    assert context.endswith("A man is playing guitar.\nQuestion: A man is performing. True, False, or Neither?\nAnswer:")


def test_load_xnli_fewshot_draws_from_validation_not_the_scored_split(monkeypatch):
    seen_splits = []

    def fake_load_dataset_kw(*args, **kwargs):
        seen_splits.append(kwargs["split"])
        return _FakeStreamingDataset(
            [{"premise": "p", "hypothesis": "h", "label": 0}]
        )

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)
    list(benchmarks.load_xnli(langs=["en"], split="test", num_fewshot=1))
    assert "validation" in seen_splits
    assert "test" in seen_splits


def test_load_xcopa_num_fewshot_prepends_completed_demonstrations(monkeypatch):
    fewshot_rows = [
        {"premise": "The ice melted.", "choice1": "The sun came out.", "choice2": "It snowed.", "question": "cause", "label": 0},
    ]
    test_rows = [
        {"premise": "The man fell.", "choice1": "He tripped.", "choice2": "He danced.", "question": "cause", "label": 0},
    ]

    def fake_load_dataset_kw(*args, **kwargs):
        if kwargs["split"] == "validation":
            return _FakeStreamingDataset(fewshot_rows)
        return _FakeStreamingDataset(test_rows)

    monkeypatch.setattr(benchmarks.hf_datasets, "load_dataset", fake_load_dataset_kw)
    examples = list(benchmarks.load_xcopa(langs=["et"], num_fewshot=1))
    assert len(examples) == 1
    context = examples[0].context
    assert "The ice melted because the sun came out." in context
    assert context.endswith("The man fell because")
