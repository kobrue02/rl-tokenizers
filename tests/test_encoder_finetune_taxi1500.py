"""Tests for encoder_finetune_taxi1500: TSV parsing (offline, hand-written
fixture) and the download step (monkeypatched requests.get -- no real
network call, matching this project's own convention for external-source
tests, e.g. test_data_prep.py's monkeypatched stream_groups)."""

import pytest

from systems.pretraining.encoder_finetune_taxi1500 import (
    download_taxi1500_split,
    load_taxi1500_tsv,
)


def test_load_taxi1500_tsv_parses_rows_and_maps_labels(tmp_path):
    path = tmp_path / "eng_train.tsv"
    path.write_text(
        "62003016\tFaith\tBy this we have come to know love.\n"
        "43013001\tGrace\tNow because he knew before the festival.\n"
        "58003017\tSin\tMoreover, with whom did God become disgusted?\n"
    )

    rows = load_taxi1500_tsv(str(path))

    assert rows == [
        {"text": "By this we have come to know love.", "label": 1},  # Faith
        {"text": "Now because he knew before the festival.", "label": 3},  # Grace
        {"text": "Moreover, with whom did God become disgusted?", "label": 4},  # Sin
    ]


def test_load_taxi1500_tsv_skips_blank_lines(tmp_path):
    path = tmp_path / "eng_train.tsv"
    path.write_text("1\tFaith\ttext one\n\n2\tSin\ttext two\n")

    rows = load_taxi1500_tsv(str(path))

    assert len(rows) == 2


def test_load_taxi1500_tsv_rejects_unrecognized_label(tmp_path):
    path = tmp_path / "eng_train.tsv"
    path.write_text("1\tNotARealLabel\tsome text\n")

    with pytest.raises(ValueError, match="unrecognized label"):
        load_taxi1500_tsv(str(path))


def test_load_taxi1500_tsv_rejects_malformed_line(tmp_path):
    path = tmp_path / "eng_train.tsv"
    path.write_text("only\ttwo_fields\n")

    with pytest.raises(ValueError, match="expected 3 tab-separated fields"):
        load_taxi1500_tsv(str(path))


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_download_taxi1500_split_fetches_the_right_url_and_writes_the_file(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse("1\tFaith\thello world\n")

    monkeypatch.setattr(
        "systems.pretraining.encoder_finetune_taxi1500.requests.get", fake_get
    )

    path = download_taxi1500_split("train", str(tmp_path))

    # Asserted against the LITERAL, live-verified URL, not
    # TAXI1500_GITHUB_RAW.format(...) -- a wrong constant would otherwise
    # pass this test by construction. Real bug this guards against: the
    # constant briefly read "eng_data/{split}.tsv" (missing the "eng_"
    # prefix on the FILENAME itself, not just the local cache path) --
    # confirmed live to 404 -- while this same test, checked only against
    # the module's own constant, kept passing the whole time.
    assert calls == ["https://raw.githubusercontent.com/cisnlp/Taxi1500/main/eng_data/eng_train.tsv"]
    assert path == str(tmp_path / "eng_train.tsv")
    assert load_taxi1500_tsv(path) == [{"text": "hello world", "label": 1}]


def test_download_taxi1500_split_is_idempotent(tmp_path, monkeypatch):
    """A second call must NOT re-fetch if the file already exists --
    matches this project's own local-cache scripts' idempotent-rerun
    convention (e.g. common/data/prepare_bible_nlp.py)."""
    call_count = {"n": 0}

    def fake_get(url, timeout):
        call_count["n"] += 1
        return _FakeResponse("1\tFaith\thello\n")

    monkeypatch.setattr(
        "systems.pretraining.encoder_finetune_taxi1500.requests.get", fake_get
    )

    download_taxi1500_split("train", str(tmp_path))
    download_taxi1500_split("train", str(tmp_path))

    assert call_count["n"] == 1
