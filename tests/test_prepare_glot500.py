"""Tests for common.data.prepare_glot500 -- the one-time local-cache
downloader for glot500 (see that module's own docstring for why this needs
to be resumable and parallelized, unlike prepare_bible_nlp/
prepare_indigenous_panel). No network access: common.data.corpora._stream_hf
is monkeypatched to a small in-memory fake per language."""

import json
import os

import pytest

from common.data import prepare_glot500 as prepare_glot500_module
from common.data.prepare_glot500 import prepare_glot500


FAKE_CORPUS = {
    "eng_Latn": ["hello world", "another english document"],
    "fra_Latn": ["bonjour le monde"],
    "deu_Latn": ["hallo welt", "noch ein dokument", "und noch eins"],
}


def _fake_stream_hf(repo_id, config, split="train"):
    for text in FAKE_CORPUS[config]:
        yield {"text": text}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(prepare_glot500_module, "_stream_hf", _fake_stream_hf)
    monkeypatch.setattr(prepare_glot500_module, "_resolve_glot500_config", lambda lang: lang)
    monkeypatch.setattr(prepare_glot500_module, "list_glot500_configs", lambda: sorted(FAKE_CORPUS))


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_writes_one_jsonl_per_language_with_correct_rows(tmp_path):
    prepare_glot500(str(tmp_path), max_workers=2)

    for lang, texts in FAKE_CORPUS.items():
        rows = _read_jsonl(tmp_path / f"{lang}.jsonl")
        assert rows == [{lang: t} for t in texts]
    assert not list(tmp_path.glob("*.jsonl.tmp"))  # no leftover temp files


def test_writes_metadata_json_summary(tmp_path):
    meta = prepare_glot500(str(tmp_path), max_workers=2)
    assert meta["eng_Latn"]["status"] == "written"
    assert meta["eng_Latn"]["num_docs"] == 2
    with open(tmp_path / "metadata.json") as f:
        assert json.load(f) == meta


def test_limit_processes_only_first_n_languages(tmp_path):
    meta = prepare_glot500(str(tmp_path), max_workers=2, limit=1)
    assert len(meta) == 1
    assert list(meta)[0] == sorted(FAKE_CORPUS)[0]


def test_langs_subset_processes_only_requested_languages(tmp_path):
    meta = prepare_glot500(str(tmp_path), langs=["deu_Latn"], max_workers=2)
    assert set(meta) == {"deu_Latn"}


def test_rerun_skips_already_prepared_languages(tmp_path):
    prepare_glot500(str(tmp_path), max_workers=2)
    eng_path = tmp_path / "eng_Latn.jsonl"
    mtime_before = os.path.getmtime(eng_path)

    meta = prepare_glot500(str(tmp_path), max_workers=2)

    assert meta["eng_Latn"]["status"] == "skipped_already_exists"
    assert os.path.getmtime(eng_path) == mtime_before  # genuinely not rewritten


def test_force_redownloads_even_when_already_prepared(tmp_path):
    prepare_glot500(str(tmp_path), max_workers=2)
    meta = prepare_glot500(str(tmp_path), max_workers=2, force=True)
    assert meta["eng_Latn"]["status"] == "written"


def test_one_language_failing_does_not_abort_the_others(tmp_path, monkeypatch):
    def flaky_stream_hf(repo_id, config, split="train"):
        if config == "fra_Latn":
            raise RuntimeError("simulated network failure")
        yield from _fake_stream_hf(repo_id, config, split=split)

    monkeypatch.setattr(prepare_glot500_module, "_stream_hf", flaky_stream_hf)

    meta = prepare_glot500(str(tmp_path), max_workers=3)

    assert meta["fra_Latn"]["status"] == "failed"
    assert "simulated network failure" in meta["fra_Latn"]["error"]
    assert meta["eng_Latn"]["status"] == "written"
    assert meta["deu_Latn"]["status"] == "written"
    assert (tmp_path / "eng_Latn.jsonl").exists()
    assert (tmp_path / "deu_Latn.jsonl").exists()
    assert not (tmp_path / "fra_Latn.jsonl").exists()
    assert "_failed" in meta and meta["_failed"] == {"fra_Latn": meta["fra_Latn"]["error"]}


def test_resumed_run_after_failure_only_redoes_the_failed_language(tmp_path, monkeypatch):
    call_log = []

    def flaky_once(repo_id, config, split="train"):
        call_log.append(config)
        if config == "fra_Latn" and call_log.count("fra_Latn") == 1:
            raise RuntimeError("simulated crash")
        yield from _fake_stream_hf(repo_id, config, split=split)

    monkeypatch.setattr(prepare_glot500_module, "_stream_hf", flaky_once)
    meta_first = prepare_glot500(str(tmp_path), max_workers=3)
    assert meta_first["fra_Latn"]["status"] == "failed"

    meta_second = prepare_glot500(str(tmp_path), max_workers=3)
    assert meta_second["eng_Latn"]["status"] == "skipped_already_exists"
    assert meta_second["deu_Latn"]["status"] == "skipped_already_exists"
    assert meta_second["fra_Latn"]["status"] == "written"
    assert _read_jsonl(tmp_path / "fra_Latn.jsonl") == [{"fra_Latn": "bonjour le monde"}]
