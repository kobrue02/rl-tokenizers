"""Tests for scripts.combine_decoder_results: flattening multiple
systems.pretraining.cli_eval --output JSON files (each covering one or more
benchmarks for one label) into one {label: {benchmark: result}} comparison."""

import json

import pytest

from scripts.combine_decoder_results import combine_decoder_results, main


def _write(tmp_path, name, record):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return str(path)


def test_flattens_a_multi_benchmark_file_under_its_label(tmp_path):
    all_bpe = _write(tmp_path, "all_bpe.json", {
        "label": "bpe", "benchmark": ["xnli", "xcopa"],
        "results": {
            "xnli": {"accuracy": 0.42, "n": 100},
            "xcopa": {"accuracy": 0.51, "n": 100},
        },
    })

    combined = combine_decoder_results([all_bpe])

    assert set(combined.keys()) == {"bpe"}
    assert combined["bpe"]["xnli"]["accuracy"] == 0.42
    assert combined["bpe"]["xcopa"]["accuracy"] == 0.51


def test_merges_same_benchmark_across_different_labels(tmp_path):
    bpe = _write(tmp_path, "all_bpe.json", {
        "label": "bpe", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.42, "n": 100}},
    })
    fanta = _write(tmp_path, "all_fanta.json", {
        "label": "fanta", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.47, "n": 100}},
    })

    combined = combine_decoder_results([bpe, fanta])

    assert combined["bpe"]["xnli"]["accuracy"] == 0.42
    assert combined["fanta"]["xnli"]["accuracy"] == 0.47


def test_merges_separate_single_benchmark_files_under_the_same_label(tmp_path):
    xnli = _write(tmp_path, "xnli_bpe.json", {
        "label": "bpe", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.42, "n": 100}},
    })
    flores = _write(tmp_path, "flores_bpe.json", {
        "label": "bpe", "benchmark": ["flores_mt"], "results": {"flores_mt": {"bleu": 3.2, "chrf": 18.5, "n": 50}},
    })

    combined = combine_decoder_results([xnli, flores])

    assert combined["bpe"]["xnli"]["accuracy"] == 0.42
    assert combined["bpe"]["flores_mt"]["bleu"] == 3.2


def test_collision_warns_and_keeps_the_later_file(tmp_path, capsys):
    first = _write(tmp_path, "xnli_bpe_v1.json", {
        "label": "bpe", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.42, "n": 100}},
    })
    second = _write(tmp_path, "xnli_bpe_v2.json", {
        "label": "bpe", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.50, "n": 100}},
    })

    combined = combine_decoder_results([first, second])

    out = capsys.readouterr().out
    assert "appears in more than one input file" in out
    assert combined["bpe"]["xnli"]["accuracy"] == 0.50


def test_record_without_results_raises_a_clear_error(tmp_path):
    bad = _write(tmp_path, "bad.json", {"label": "bpe", "benchmark": ["xnli"]})

    with pytest.raises(ValueError, match="no 'results' key"):
        combine_decoder_results([bad])


def test_main_writes_the_combined_json_to_output(tmp_path):
    all_bpe = _write(tmp_path, "all_bpe.json", {
        "label": "bpe", "benchmark": ["xnli"], "results": {"xnli": {"accuracy": 0.42, "n": 100}},
    })
    output_path = tmp_path / "combined.json"

    main(["--input", all_bpe, "--output", str(output_path)])

    with open(output_path) as f:
        combined = json.load(f)
    assert combined["bpe"]["xnli"]["accuracy"] == 0.42
