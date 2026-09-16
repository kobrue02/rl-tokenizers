"""Fast, offline unit tests for lm_eval_adapter.py's own logic that doesn't
need a real model/network access: _lang_from_task_name's prefix parsing
(confirmed against each real task family's own naming convention on
github.com/EleutherAI/lm-evaluation-harness directly, not assumed) and
ThesisLM's constructor guard. See test_cli_lm_eval.py for the real,
network-dependent end-to-end integration test."""

import pytest

from systems.pretraining.lm_eval_adapter import ThesisLM, _lang_from_task_name


def test_lang_from_task_name_xnli_and_xcopa():
    assert _lang_from_task_name("xnli_sw") == "sw"
    assert _lang_from_task_name("xnli_en") == "en"
    assert _lang_from_task_name("xcopa_et") == "et"


def test_lang_from_task_name_xstorycloze():
    assert _lang_from_task_name("xstorycloze_ar") == "ar"


def test_lang_from_task_name_lambada_openai_mt():
    assert _lang_from_task_name("lambada_openai_mt_de") == "de"
    assert _lang_from_task_name("lambada_openai_mt_en") == "en"


def test_lang_from_task_name_global_piqa_cloze_variants():
    assert _lang_from_task_name("global_piqa_nonparallel_cloze_als_latn") == "als_latn"
    assert _lang_from_task_name("global_piqa_parallel_cloze_amh_ethi") == "amh_ethi"
    # A three-part region-suffixed dialect stem -- see lm_eval_adapter.py's
    # own comment on this being a real, narrow, pre-existing limitation of
    # tokenizer_adapter.py's OWN script resolver for such stems (not
    # something _lang_from_task_name itself needs to disambiguate; it just
    # needs to extract the whole trailing stem correctly, which it does).
    assert _lang_from_task_name("global_piqa_nonparallel_cloze_apc_arab_jord") == "apc_arab_jord"


def test_lang_from_task_name_unknown_prefix_returns_none():
    # blimp's own paradigm-named tasks (e.g. "adjunct_island") -- no known
    # prefix, correctly falls back to None (TokenizerAdapter.encode's own
    # lang=None default), not a wrong guess.
    assert _lang_from_task_name("blimp_adjunct_island") is None
    assert _lang_from_task_name("adjunct_island") is None


def test_lang_from_task_name_empty_or_none_returns_none():
    assert _lang_from_task_name(None) is None
    assert _lang_from_task_name("") is None


def test_thesis_lm_bare_constructor_requires_model_and_adapter():
    with pytest.raises(ValueError):
        ThesisLM()
    with pytest.raises(ValueError):
        ThesisLM(model=object())
