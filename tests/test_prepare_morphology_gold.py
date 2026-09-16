"""Tests for common.data.prepare_morphology_gold. Network-fetch functions
(_prepare_morphynet_language etc.) are NOT mocked/tested here -- they're
exercised for real by a live run reported separately; these tests cover
the pure logic (alignment heuristic, jsonl round-trip, dispatch/error
handling) that can be verified without a network call."""

import json
import os

from common.data.prepare_morphology_gold import (
    _MORPHYNET_LANGS,
    _SIGMORPHON_LANGS,
    _UNIMORPH_INDIGENOUS_LANGS,
    _align_unimorph_lemma_form,
    load_morphology_gold,
    prepare_morphology_gold,
)


def test_morphynet_langs_uses_mon_not_khk_and_excludes_hbs():
    """See module docstring: MorphyNet's own repo directory is "mon", not
    "khk", and includes an "hbs" directory this project doesn't use."""
    assert "mon" in _MORPHYNET_LANGS
    assert "khk" not in _MORPHYNET_LANGS
    assert _MORPHYNET_LANGS["mon"] == "khk_Cyrl"
    assert "hbs" not in _MORPHYNET_LANGS
    assert len(_MORPHYNET_LANGS) == 14


def test_sigmorphon_langs_are_the_three_morphbpe_comparable_languages():
    assert set(_SIGMORPHON_LANGS.values()) == {"eng_Latn", "rus_Cyrl", "hun_Latn"}


def test_unimorph_indigenous_langs_use_bare_codes():
    assert _UNIMORPH_INDIGENOUS_LANGS == {"aym": "aym", "cni": "cni", "shp": "shp"}


def test_align_unimorph_identical_lemma_and_form_is_one_morph():
    assert _align_unimorph_lemma_form("cat", "cat") == ["cat"]


def test_align_unimorph_prefix_and_suffix_match():
    # lemma "walk", form "walked": common prefix "walk", no common suffix
    # (would overlap the prefix), middle/suffix split gives stem + affix.
    morphs = _align_unimorph_lemma_form("walk", "walked")
    assert "".join(morphs) == "walked"
    assert morphs[0] == "walk"
    assert morphs[-1] == "ed"


def test_align_unimorph_common_prefix_and_suffix_both_present():
    # lemma "ab", form "aXXb": prefix "a", suffix "b", middle "XX".
    morphs = _align_unimorph_lemma_form("ab", "aXXb")
    assert morphs == ["a", "XX", "b"]


def test_align_unimorph_no_overlap_falls_back_to_whole_form():
    morphs = _align_unimorph_lemma_form("xyz", "abc")
    assert morphs == ["abc"]


def test_load_morphology_gold_missing_dir_returns_empty():
    assert load_morphology_gold("/nonexistent/path/xyz123") == {}


def test_load_morphology_gold_round_trips_written_jsonl(tmp_path):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    with open(gold_dir / "deu_Latn.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"word": "holmi", "morphs": ["hol", "mi"], "source": "morphynet"}) + "\n")
        f.write(json.dumps({"word": "katze", "morphs": ["katz", "e"], "source": "morphynet"}) + "\n")

    loaded = load_morphology_gold(str(gold_dir))

    assert set(loaded) == {"deu_Latn"}
    assert loaded["deu_Latn"]["holmi"] == (["hol", "mi"], "morphynet")
    assert loaded["deu_Latn"]["katze"] == (["katz", "e"], "morphynet")


def test_prepare_morphology_gold_rejects_unknown_source(tmp_path):
    try:
        prepare_morphology_gold(str(tmp_path), sources=["not_a_real_source"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_real_source" in str(e)


def test_prepare_morphology_gold_sigmorphon_overrides_morphynet_for_shared_langs(monkeypatch, tmp_path):
    """eng_Latn gets produced by BOTH morphynet and sigmorphon fakes here --
    confirm sigmorphon's version (applied second) wins entirely, not a
    per-word merge."""
    import common.data.prepare_morphology_gold as mod

    def fake_morphynet(request_timeout):
        return {"eng_Latn": {"cats": (["cat", "s"], "morphynet")}}, {}

    def fake_sigmorphon(request_timeout):
        return {"eng_Latn": {"dogs": (["dog", "s"], "sigmorphon2022segmentation")}}, {}

    def fake_unimorph(request_timeout):
        return {}, {}

    monkeypatch.setattr(mod, "_prepare_morphynet", fake_morphynet)
    monkeypatch.setattr(mod, "_prepare_sigmorphon_segmentation", fake_sigmorphon)
    monkeypatch.setattr(mod, "_prepare_unimorph_indigenous", fake_unimorph)
    monkeypatch.setitem(mod._SOURCE_PREPARERS, "morphynet", fake_morphynet)
    monkeypatch.setitem(mod._SOURCE_PREPARERS, "sigmorphon", fake_sigmorphon)
    monkeypatch.setitem(mod._SOURCE_PREPARERS, "unimorph", fake_unimorph)

    summary = prepare_morphology_gold(str(tmp_path))

    assert summary["languages"]["eng_Latn"]["source"] == "sigmorphon2022segmentation"
    loaded = load_morphology_gold(str(tmp_path))
    assert set(loaded["eng_Latn"]) == {"dogs"}


def test_prepare_morphology_gold_collects_errors_without_aborting(monkeypatch, tmp_path):
    import common.data.prepare_morphology_gold as mod

    def failing_morphynet(request_timeout):
        return {}, {"eng": "boom"}

    def fake_sigmorphon(request_timeout):
        return {"rus_Cyrl": {"привет": (["привет"], "sigmorphon2022segmentation")}}, {}

    def fake_unimorph(request_timeout):
        return {}, {}

    monkeypatch.setitem(mod._SOURCE_PREPARERS, "morphynet", failing_morphynet)
    monkeypatch.setitem(mod._SOURCE_PREPARERS, "sigmorphon", fake_sigmorphon)
    monkeypatch.setitem(mod._SOURCE_PREPARERS, "unimorph", fake_unimorph)

    summary = prepare_morphology_gold(str(tmp_path))

    assert summary["errors"]["morphynet"] == {"eng": "boom"}
    assert "rus_Cyrl" in summary["languages"]
