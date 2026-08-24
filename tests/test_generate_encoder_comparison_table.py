"""Tests for scripts.generate_encoder_comparison_table: the headline-metric
extraction (single-value and --eval-langs=all-style multi-language
averaging), the summary/detailed table shapes, and missing-cell handling."""

import json

import pytest

from scripts.combine_encoder_results import main as combine_main
from scripts.generate_encoder_comparison_table import (
    _headline_value,
    build_detailed_table,
    build_summary_table,
    main,
    render_markdown_table,
)


def test_headline_value_exact_key_lookup():
    assert _headline_value({"pseudoperplexity": 12.3}, "pseudoperplexity") == 12.3


def test_headline_value_averages_matching_suffix_keys_single_language():
    # A single --eval-lang run: one eval_f1 key -- "average of one" == that value.
    assert _headline_value({"eval_f1": 0.8, "eval_loss": 0.1}, "_f1") == 0.8


def test_headline_value_averages_matching_suffix_keys_multi_language():
    # An --eval-langs=all sweep: eval_deu_f1/eval_fra_f1/eval_ita_f1 -- the
    # actual "how did this tokenizer do across every language" summary.
    result = {"eval_deu_f1": 0.6, "eval_fra_f1": 0.8, "eval_ita_f1": 1.0, "eval_deu_loss": 0.3}
    assert _headline_value(result, "_f1") == pytest.approx(0.8)


def test_headline_value_returns_none_when_nothing_matches():
    assert _headline_value({"eval_loss": 0.1}, "_f1") is None


def test_build_summary_table_shows_dashes_for_missing_benchmarks():
    combined = {
        "bpe": {"pppl": {"result": {"pseudoperplexity": 12.3}}},
        "fanta": {"retrieval": {"result": {"top10_accuracy": 0.5}}},
    }

    header, rows = build_summary_table(combined)

    assert header == ["label", "pppl", "retrieval"]
    rows_by_label = {row[0]: row[1:] for row in rows}
    assert rows_by_label["bpe"] == ["12.3000", "--"]
    assert rows_by_label["fanta"] == ["--", "0.5000"]


def test_build_summary_table_averages_ner_across_eval_langs_all():
    combined = {
        "bpe": {"ner": {"result": {"eval_deu_f1": 0.6, "eval_fra_f1": 0.8}}},
    }

    header, rows = build_summary_table(combined)

    assert header == ["label", "ner"]
    assert rows[0] == ["bpe", "0.7000"]


def test_build_detailed_table_includes_every_raw_metric_key():
    combined = {
        "bpe": {"retrieval": {"result": {"top1_accuracy": 0.3, "top10_accuracy": 0.5}}},
    }

    header, rows = build_detailed_table(combined)

    # Plain lexicographic sort, not numeric -- "top10_accuracy" sorts before
    # "top1_accuracy" ('0' < '_' as characters), which is fine (order just
    # needs to be deterministic, not numerically sensible).
    assert set(header) == {"label", "retrieval.top1_accuracy", "retrieval.top10_accuracy"}
    row = dict(zip(header, rows[0]))
    assert row["retrieval.top1_accuracy"] == "0.3000"
    assert row["retrieval.top10_accuracy"] == "0.5000"


def test_build_detailed_table_missing_metric_for_one_label_is_a_dash():
    combined = {
        "bpe": {"pppl": {"result": {"pseudoperplexity": 12.3}}},
        "fanta": {"pppl": {"result": {}}},  # ran but this particular metric key absent
    }

    header, rows = build_detailed_table(combined)

    rows_by_label = {row[0]: row[1:] for row in rows}
    assert rows_by_label["fanta"] == ["--"]


def test_render_markdown_table_produces_a_pipe_table():
    md = render_markdown_table(["label", "pppl"], [["bpe", "12.3000"]])
    lines = md.splitlines()
    assert lines[0] == "| label | pppl |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| bpe | 12.3000 |"


def test_full_pipeline_from_two_result_files_to_markdown_report(tmp_path):
    """The actual end-to-end use case: two encoder_cli_eval --output files
    for two tokenizers -> combine_encoder_results -> this script -- a real
    smoke test that the two scripts' JSON shapes actually fit together."""
    pppl_bpe = tmp_path / "pppl_bpe.json"
    pppl_bpe.write_text(json.dumps({
        "label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3},
    }))
    pppl_fanta = tmp_path / "pppl_fanta.json"
    pppl_fanta.write_text(json.dumps({
        "label": "fanta", "benchmark": "pppl", "result": {"pseudoperplexity": 15.7},
    }))
    combined_path = tmp_path / "combined.json"
    combine_main(["--input", str(pppl_bpe), str(pppl_fanta), "--output", str(combined_path)])

    report_path = tmp_path / "report.md"
    main(["--input", str(combined_path), "--output", str(report_path)])

    report = report_path.read_text()
    assert "## Summary" in report
    assert "## Detailed" in report
    assert "bpe" in report and "fanta" in report
    assert "12.3000" in report and "15.7000" in report
