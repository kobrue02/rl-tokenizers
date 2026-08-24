"""Tests for scripts.combine_encoder_results: the nested (label, benchmark/
task) merge across multiple encoder_cli_eval/encoder_cli_finetune --output
JSON files."""

import json

import pytest

from scripts.combine_encoder_results import combine_encoder_results, main


def _write(tmp_path, name, record):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return str(path)


def test_merges_different_benchmarks_under_the_same_label(tmp_path):
    pppl_bpe = _write(tmp_path, "pppl_bpe.json", {
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3},
    })
    retrieval_bpe = _write(tmp_path, "retrieval_bpe.json", {
        "label": "bpe", "benchmark": "retrieval", "result": {"top10_accuracy": 0.5},
    })

    combined = combine_encoder_results([pppl_bpe, retrieval_bpe])

    assert set(combined.keys()) == {"bpe"}
    assert combined["bpe"]["pppl"]["result"]["pseudoperplexity"] == 12.3
    assert combined["bpe"]["retrieval"]["result"]["top10_accuracy"] == 0.5


def test_merges_same_benchmark_across_different_labels(tmp_path):
    pppl_bpe = _write(tmp_path, "pppl_bpe.json", {
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3},
    })
    pppl_fanta = _write(tmp_path, "pppl_fanta.json", {
        "label": "fanta", "benchmark": "pppl", "result": {"pseudoperplexity": 15.7},
    })

    combined = combine_encoder_results([pppl_bpe, pppl_fanta])

    assert combined["bpe"]["pppl"]["result"]["pseudoperplexity"] == 12.3
    assert combined["fanta"]["pppl"]["result"]["pseudoperplexity"] == 15.7


def test_finetune_records_use_task_not_benchmark_as_the_key(tmp_path):
    ner_bpe = _write(tmp_path, "ner_bpe.json", {
        "label": "bpe", "task": "ner", "result": {"eval_f1": 0.8},
    })

    combined = combine_encoder_results([ner_bpe])

    assert combined["bpe"]["ner"]["result"]["eval_f1"] == 0.8


def test_collision_warns_and_keeps_the_later_file(tmp_path, capsys):
    first = _write(tmp_path, "pppl_bpe_v1.json", {
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3},
    })
    second = _write(tmp_path, "pppl_bpe_v2.json", {
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 9.9},
    })

    combined = combine_encoder_results([first, second])

    out = capsys.readouterr().out
    assert "appears in more than one input file" in out
    assert combined["bpe"]["pppl"]["result"]["pseudoperplexity"] == 9.9


def test_record_without_benchmark_or_task_raises_a_clear_error(tmp_path):
    bad = _write(tmp_path, "bad.json", {"label": "bpe", "result": {"x": 1}})

    with pytest.raises(ValueError, match="neither 'benchmark' nor 'task'"):
        combine_encoder_results([bad])


def test_main_writes_the_combined_json_to_output(tmp_path):
    pppl_bpe = _write(tmp_path, "pppl_bpe.json", {
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3},
    })
    output_path = tmp_path / "combined.json"

    main(["--input", pppl_bpe, "--output", str(output_path)])

    with open(output_path) as f:
        combined = json.load(f)
    assert combined["bpe"]["pppl"]["result"]["pseudoperplexity"] == 12.3
