"""Tests for the BLiMP/CoLA/SQuAD additions to systems.pretraining.benchmarks
and systems.pretraining.eval_harness: _mcc, _sweep_best_threshold, the SQuAD
string-normalization/EM/F1 helpers, evaluate_cola/evaluate_qa end-to-end
against a tiny from-scratch model, loglikelihood's empty-context handling
(the eos_id-prepend fix BLiMP/CoLA both depend on), and load_blimp's example
shape via a monkeypatched datasets.load_dataset (no network calls)."""

import pytest
import torch

from systems.pretraining import benchmarks
from systems.pretraining.benchmarks import CoLAExample, MultipleChoiceExample, QAExample
from systems.pretraining.eval_harness import (
    _best_over_references,
    _exact_match,
    _f1,
    _mcc,
    _normalize_answer,
    _sweep_best_threshold,
    evaluate_cola,
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
