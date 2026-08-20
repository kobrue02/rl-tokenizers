"""Tests for common.data.corpora's glot500 local-disk-cache read path (see
common/data/prepare_glot500.py's own module docstring for why glot500
moved off live HF streaming -- confirmed live to be the actual bottleneck
of a real glot500-scale pretraining data prep). Mirrors the existing
bible_nlp/indigenous_panel local-cache convention: no network access here,
purely reading hand-built JSONL fixtures."""

import json

import pytest

from common.data.corpora import (
    _stream_glot500_local_single,
    list_glot500_local_langs,
    stream_groups,
)


def _write_lang_file(tmp_path, lang, texts):
    path = tmp_path / f"{lang}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({lang: text}) + "\n")
    return path


def test_list_glot500_local_langs_empty_when_dir_missing(tmp_path):
    assert list_glot500_local_langs(str(tmp_path / "does_not_exist")) == []


def test_list_glot500_local_langs_lists_only_jsonl_files(tmp_path):
    _write_lang_file(tmp_path, "eng_Latn", ["hello"])
    _write_lang_file(tmp_path, "fra_Latn", ["bonjour"])
    (tmp_path / "metadata.json").write_text("{}")  # must NOT be counted as a language
    (tmp_path / "eng_Latn.jsonl.tmp").write_text("")  # in-progress write, must NOT be counted

    assert list_glot500_local_langs(str(tmp_path)) == ["eng_Latn", "fra_Latn"]


def test_stream_glot500_local_single_raises_with_prepare_instruction_when_missing(tmp_path):
    with pytest.raises(ValueError, match="prepare_glot500"):
        list(_stream_glot500_local_single("eng_Latn", output_dir=str(tmp_path)))


def test_stream_glot500_local_single_yields_rows_in_file_order(tmp_path):
    _write_lang_file(tmp_path, "eng_Latn", ["first", "second", "third"])
    rows = list(_stream_glot500_local_single("eng_Latn", output_dir=str(tmp_path)))
    assert rows == [{"eng_Latn": "first"}, {"eng_Latn": "second"}, {"eng_Latn": "third"}]


def test_stream_glot500_local_single_skips_blank_lines(tmp_path):
    path = tmp_path / "eng_Latn.jsonl"
    path.write_text('{"eng_Latn": "a"}\n\n{"eng_Latn": "b"}\n')
    rows = list(_stream_glot500_local_single("eng_Latn", output_dir=str(tmp_path)))
    assert rows == [{"eng_Latn": "a"}, {"eng_Latn": "b"}]


def test_stream_groups_glot500_single_language_reads_local_cache(tmp_path):
    _write_lang_file(tmp_path, "eng_Latn", ["doc one", "doc two"])
    rows = list(stream_groups("glot500", langs=["eng_Latn"], config=str(tmp_path)))
    assert rows == [{"eng_Latn": "doc one"}, {"eng_Latn": "doc two"}]


def test_stream_groups_glot500_multi_language_round_robins(tmp_path):
    _write_lang_file(tmp_path, "eng_Latn", ["e1", "e2"])
    _write_lang_file(tmp_path, "fra_Latn", ["f1"])
    rows = list(stream_groups("glot500", langs=["eng_Latn", "fra_Latn"], config=str(tmp_path)))
    # _round_robin interleaves one item at a time per active iterator, dropping
    # an iterator once it's exhausted -- eng_Latn (2 items) outlives fra_Latn (1).
    assert rows == [{"eng_Latn": "e1"}, {"fra_Latn": "f1"}, {"eng_Latn": "e2"}]


def test_stream_groups_glot500_all_reads_whatever_is_actually_present(tmp_path):
    """langs="all" (or omitted) must use list_glot500_local_langs (what's
    ACTUALLY on disk) rather than the full ~411-language HF manifest --
    critical for a partial/in-progress cache (see prepare_glot500.py's own
    RESUMABLE docstring section) to still be usable."""
    _write_lang_file(tmp_path, "eng_Latn", ["only this one language prepared so far"])
    rows = list(stream_groups("glot500", langs="all", config=str(tmp_path)))
    assert rows == [{"eng_Latn": "only this one language prepared so far"}]
