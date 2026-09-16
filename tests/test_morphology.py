"""Tests for common.eval.morphology's Morphological Edit Distance and
Morphological Consistency F1 (Asgari et al.'s "MorphBPE", arXiv:2502.00894,
Sec 3.3(ii)/(iii) -- both prose-only in that paper, no formulas, so these
tests lock in THIS project's own concrete, documented formalization choices,
not a verified reproduction of undisclosed paper internals)."""

from common.eval.morphology import (
    evaluate_morphological_segmentation,
    morphological_consistency_f1,
    morphological_edit_distance,
)


def test_med_identical_sequences_is_zero():
    assert morphological_edit_distance([b"un", b"happy"], [b"un", b"happy"]) == 0


def test_med_one_substitution():
    assert morphological_edit_distance([b"un", b"happy"], [b"un", b"happi"]) == 1


def test_med_unnormalized_not_divided_by_length():
    """The paper explicitly retains MED in raw form ("we retain its raw form
    to provide a clearer indication of the average number of edits
    required") rather than normalizing by morpheme count -- confirm this
    function does NOT divide by len(gold_morphs)."""
    gold = [b"a", b"b", b"c", b"d"]
    predicted = [b"x", b"y", b"z", b"w"]  # 4 substitutions
    assert morphological_edit_distance(gold, predicted) == 4  # not 4/4=1.0


def test_med_merged_token_costs_more_than_one_edit():
    # gold splits "undo" into two morphs; a tokenizer that doesn't split it at
    # all needs 2 edits (delete "un", substitute "do"->"undo", or equivalent).
    assert morphological_edit_distance([b"un", b"do"], [b"undo"]) == 2


def test_consistency_f1_perfect_segmentation_scores_one():
    """Every word's predicted spans match its gold morphs exactly -- shared
    morphemes and shared tokens coincide perfectly, so precision=recall=f1=1."""
    data = {
        "undo": ([b"un", b"do"], [b"un", b"do"]),
        "unhappy": ([b"un", b"happy"], [b"un", b"happy"]),
        "happy": ([b"happy"], [b"happy"]),
    }
    result = morphological_consistency_f1(data, n_bootstrap=50, seed=0)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["n_pairs"] == 3


def test_consistency_f1_never_splitting_shared_morphemes_scores_zero():
    """"unhappy" is never split, so it shares no token with either "undo"
    (which shares gold morph "un") or "happy" (which shares gold morph
    "happy") -- recall must be 0 (morpheme-shared pairs never token-shared),
    and precision is 0/0 since no pair is ever token-shared at all."""
    data = {
        "undo": ([b"un", b"do"], [b"un", b"do"]),
        "unhappy": ([b"un", b"happy"], [b"unhappy"]),  # not split
        "happy": ([b"happy"], [b"happy"]),
    }
    result = morphological_consistency_f1(data, n_bootstrap=50, seed=0)
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_consistency_f1_fewer_than_two_words_returns_zeros():
    result = morphological_consistency_f1({"solo": ([b"solo"], [b"solo"])})
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f1_ci_low": 0.0, "f1_ci_high": 0.0, "n_pairs": 0}


def test_consistency_f1_ci_bounds_contain_point_estimate():
    data = {
        "undo": ([b"un", b"do"], [b"un", b"do"]),
        "unhappy": ([b"un", b"happy"], [b"unhappy"]),
        "happy": ([b"happy"], [b"happy"]),
        "unfair": ([b"un", b"fair"], [b"un", b"fair"]),
    }
    result = morphological_consistency_f1(data, n_bootstrap=200, seed=0)
    assert result["f1_ci_low"] <= result["f1"] <= result["f1_ci_high"]


def test_consistency_f1_deterministic_given_seed():
    data = {
        "undo": ([b"un", b"do"], [b"un", b"do"]),
        "unhappy": ([b"un", b"happy"], [b"unhappy"]),
        "happy": ([b"happy"], [b"happy"]),
        "unfair": ([b"un", b"fair"], [b"un", b"fair"]),
    }
    a = morphological_consistency_f1(data, n_bootstrap=100, seed=42)
    b = morphological_consistency_f1(data, n_bootstrap=100, seed=42)
    assert a == b


def test_evaluate_morphological_segmentation_skips_language_without_induce_fn():
    gold_words_by_lang = {
        "deu_Latn": {"holmi": (["hol", "mi"], "morphynet")},
        "xyz_Latn": {"foo": (["foo"], "morphynet")},  # no induce_fn for this one
    }
    induce_fn_by_lang = {"deu_Latn": lambda raw: [raw]}  # trivial: whole word, one span
    results = evaluate_morphological_segmentation(induce_fn_by_lang, gold_words_by_lang)
    assert set(results) == {"deu_Latn"}


def test_evaluate_morphological_segmentation_reports_med_and_source():
    gold_words_by_lang = {
        "deu_Latn": {
            "holmi": (["hol", "mi"], "morphynet"),
            "katze": (["katz", "e"], "morphynet"),
        },
    }

    def induce_fn(raw):
        # A tokenizer that never splits -- always returns the word whole.
        return [raw]

    results = evaluate_morphological_segmentation({"deu_Latn": induce_fn}, gold_words_by_lang)
    assert results["deu_Latn"]["source"] == "morphynet"
    assert results["deu_Latn"]["n_words"] == 2
    # Each word needed exactly 2 edits (2 gold morphs -> 1 predicted span: one
    # substitution + one deletion), so mean MED == 2.0.
    assert results["deu_Latn"]["med"] == 2.0
