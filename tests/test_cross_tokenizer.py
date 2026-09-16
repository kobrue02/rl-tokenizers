"""Tests for the two real bugs found in common.eval.cross_tokenizer.
evaluate_on_groups on 2026-09-16 (see that function's own docstring):

1. renyi_efficiency's vocab_size normalization was never threaded through --
   it silently fell back to counting distinct token TYPES observed in the
   eval sample, not the tokenizer's true designed vocabulary size.
2. gini was computed from renyi (per-language vocabulary-usage efficiency)
   instead of token_parity (per-language token COST, what Foroutan et al.'s
   own Gini-coefficient metric actually measures) -- a materially different
   quantity.
"""

from common.eval.cross_tokenizer import evaluate_on_groups, strip_token_freq


def _fixed_spans_induce_fn(spans_by_raw):
    """spans_by_raw: {bytes: [span, ...]} -- a deterministic stand-in for a
    real induce_spans callable, keyed by the exact raw bytes evaluate_on_groups
    will pass in."""
    def induce(raw):
        return spans_by_raw[raw]
    return induce


def test_vocab_size_is_threaded_into_renyi_efficiency():
    # Two distinct token types, uneven frequency -- non-trivial entropy, so
    # the normalization denominator actually changes the reported value.
    raw = "aaab".encode("utf-8")
    induce_fn_by_lang = {"eng": _fixed_spans_induce_fn({raw: [b"a", b"a", b"a", b"b"]})}
    eval_groups = [{"eng": "aaab"}]

    no_vocab_size = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    with_vocab_size = evaluate_on_groups(induce_fn_by_lang, eval_groups, vocab_size=1000)

    # Without vocab_size: normalizes by the 2 distinct types observed here.
    # With vocab_size=1000: normalizes by the tokenizer's real (much larger)
    # vocabulary -- a materially different, and much smaller, ratio.
    assert no_vocab_size["renyi"]["eng"] != with_vocab_size["renyi"]["eng"]
    assert with_vocab_size["renyi"]["eng"] < no_vocab_size["renyi"]["eng"]


def test_gini_is_derived_from_token_parity_not_renyi():
    """eng and xyz produce IDENTICAL (zero-entropy, single-token-type) span
    distributions -- their renyi values are equal (both 0.0), so a
    renyi-based gini would be exactly 0.0. But xyz needs 2x as many tokens
    as eng for equivalent content, so token_parity differs (1.0 vs 2.0),
    and a token_parity-based gini must be nonzero. This is the exact
    scenario that would have silently passed under the old (buggy)
    renyi-based gini."""
    eng_raw = "aaaa".encode("utf-8")
    xyz_raw = "aaaaaaaa".encode("utf-8")
    induce_fn_by_lang = {
        "eng": _fixed_spans_induce_fn({eng_raw: [b"a"] * 4}),
        "xyz": _fixed_spans_induce_fn({xyz_raw: [b"a"] * 8}),
    }
    eval_groups = [{"eng": "aaaa", "xyz": "aaaaaaaa"}]

    results = evaluate_on_groups(induce_fn_by_lang, eval_groups, anchor_lang="eng")

    assert results["renyi"]["eng"] == results["renyi"]["xyz"] == 0.0
    assert results["token_parity"]["eng"] == 1.0
    assert results["token_parity"]["xyz"] == 2.0
    assert results["gini"] > 0.0, "gini must reflect the token_parity disparity, not the equal renyi values"


def test_gini_dropped_from_indigenous_panel_combined_but_present_per_anchor():
    """gini is now anchor-dependent (derived from token_parity), so it can't
    be meaningfully pooled across the indigenous panel's mixed anchors the
    way compression/fertility/renyi can -- must be dropped from "combined"
    (like token_parity itself already was) and present per-anchor instead."""
    from common.eval.cross_tokenizer import evaluate_on_indigenous_panel

    en_raw = "ab cd".encode("utf-8")
    crk_raw = "ab".encode("utf-8")
    induce_fn_by_lang = {
        "en": _fixed_spans_induce_fn({en_raw: [b"a", b"b"]}),
        "crk": _fixed_spans_induce_fn({crk_raw: [b"a"]}),
    }
    eval_groups = [{"en": "ab cd", "crk": "ab"}]

    results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)

    assert "gini" not in results["combined"]
    assert "gini" in results["token_parity_by_anchor"]["en"]


def test_strip_token_freq_preserves_morphology_key_for_indigenous_panel():
    """run_eval_cli's own --morphology-gold-dir handling adds a top-level
    "morphology" key AFTER evaluate_on_indigenous_panel returns -- the
    indigenous_panel branch of strip_token_freq reconstructs a fixed-key dict
    from scratch, which would otherwise silently drop it (unlike the
    non-indigenous branch's dict comprehension, which passes any non-
    "token_freq" key through unchanged)."""
    results = {
        "combined": {"avg_compression": 2.0, "token_freq": {"en": {}}},
        "token_parity_by_anchor": {"en": {"token_parity": {}, "token_freq": {"en": {}}}},
        "morphology_spread": {"fertility_spread": 1.0, "compression_spread": 1.0},
        "morphology": {"deu_Latn": {"med": 0.5, "consistency_f1": 0.8}},
    }
    stripped = strip_token_freq(results, is_indigenous_panel=True)
    assert stripped["morphology"] == {"deu_Latn": {"med": 0.5, "consistency_f1": 0.8}}


def test_strip_token_freq_omits_morphology_key_when_absent():
    results = {
        "combined": {"avg_compression": 2.0, "token_freq": {}},
        "token_parity_by_anchor": {},
        "morphology_spread": {"fertility_spread": 1.0, "compression_spread": 1.0},
    }
    stripped = strip_token_freq(results, is_indigenous_panel=True)
    assert "morphology" not in stripped
