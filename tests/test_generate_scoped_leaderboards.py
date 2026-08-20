"""Tests for scripts.generate_scoped_leaderboards (the HF-frontier-only and
"our work vs other approaches" leaderboard figures) and the single-column
mode gen_spread_leaderboard_tex gained to support them -- see that
function's own docstring for why a small, wide-value-range row set can't
just reuse the existing two-column split (each column would get its own
independently auto-scaled x-axis, making bar-length comparisons across the
split misleading)."""

import json

import pytest

from scripts.generate_scoped_leaderboards import (
    generate_hf_frontier_leaderboard,
    generate_our_work_vs_other_approaches_leaderboard,
)
from scripts.generate_tikz_figures import compute_families, gen_spread_leaderboard_tex


def _fake_row(name, spread):
    return {
        "name": name, "short": name, "family": "Other",
        "avg_compression": 2.0, "gini": 0.05, "spread": spread, "token_parity": {},
    }


def test_gen_spread_leaderboard_single_column_below_threshold(tmp_path):
    rows = [_fake_row(f"m{i}", float(i)) for i in range(5)]
    for i, r in enumerate(rows):
        r["idx"] = i
    families = compute_families(rows)
    tex = gen_spread_leaderboard_tex(rows, families, str(tmp_path), min_rows_for_two_columns=12)
    assert tex.count(r"\begin{tikzpicture}") == 1
    assert tex.count(r"\end{tikzpicture}") == 1


def test_gen_spread_leaderboard_two_columns_at_or_above_threshold(tmp_path):
    rows = [_fake_row(f"m{i}", float(i)) for i in range(12)]
    for i, r in enumerate(rows):
        r["idx"] = i
    families = compute_families(rows)
    tex = gen_spread_leaderboard_tex(rows, families, str(tmp_path), min_rows_for_two_columns=12)
    assert tex.count(r"\begin{tikzpicture}") == 2
    assert tex.count(r"\end{tikzpicture}") == 2


def test_gen_spread_leaderboard_single_column_shares_one_xaxis_scale(tmp_path):
    """The actual bug this was written to fix: with two columns, a small,
    skewed row set (e.g. one huge outlier) gets independently auto-scaled
    x-axes per column -- confirm the single-column output has only ONE axis
    block at all (trivially, one shared scale) by counting \\begin{axis}."""
    rows = [_fake_row("small_a", 4.0), _fake_row("small_b", 5.0), _fake_row("huge_outlier", 23.0)]
    for i, r in enumerate(rows):
        r["idx"] = i
    families = compute_families(rows)
    tex = gen_spread_leaderboard_tex(rows, families, str(tmp_path), min_rows_for_two_columns=12)
    assert tex.count(r"\begin{axis}") == 1


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _fake_model_entry(spread):
    return {"avg_compression": 2.0, "gini": 0.05, "token_parity_spread": spread, "token_parity": {}}


def test_generate_hf_frontier_leaderboard_filters_to_hf_keys_only(tmp_path):
    """extra_keys=set() isolates the BASE filtering behavior from the
    default claude-opus-5 inclusion, tested separately below."""
    all_results = tmp_path / "all.json"
    hf_keys = tmp_path / "hf_frontier.json"
    _write_json(all_results, {
        "gpt2": _fake_model_entry(5.0),
        "llama": _fake_model_entry(6.0),
        "fanta": _fake_model_entry(8.0),  # NOT an hf_frontier key -- must be excluded
    })
    _write_json(hf_keys, {"gpt2": {}, "llama": {}})

    out_dir = tmp_path / "out"
    rows, families = generate_hf_frontier_leaderboard(str(all_results), str(hf_keys), str(out_dir), extra_keys=set())
    assert {r["name"] for r in rows} == {"gpt2", "llama"}
    assert (out_dir / "fig_spread_leaderboard_hf_frontier.tex").exists()
    assert not (out_dir / "_filtered_hf_frontier.json").exists()  # temp file cleaned up


def test_generate_hf_frontier_leaderboard_includes_claude_by_default(tmp_path):
    """Regression test: claude-opus-5 isn't hosted on the HF Hub, so it's
    absent from results/hf_frontier_comparison.json's own key set -- but it's
    unambiguously a frontier model's tokenizer and must still appear in a
    "broad frontier comparison" by default, not be silently dropped."""
    all_results = tmp_path / "all.json"
    hf_keys = tmp_path / "hf_frontier.json"
    _write_json(all_results, {
        "gpt2": _fake_model_entry(5.0),
        "claude-opus-5": _fake_model_entry(8.0),
        "fanta": _fake_model_entry(8.0),
    })
    _write_json(hf_keys, {"gpt2": {}})  # claude-opus-5 deliberately absent here

    rows, families = generate_hf_frontier_leaderboard(str(all_results), str(hf_keys), str(tmp_path / "out"))
    assert {r["name"] for r in rows} == {"gpt2", "claude-opus-5"}


def test_generate_hf_frontier_leaderboard_raises_on_missing_key(tmp_path):
    all_results = tmp_path / "all.json"
    hf_keys = tmp_path / "hf_frontier.json"
    _write_json(all_results, {"gpt2": _fake_model_entry(5.0)})
    _write_json(hf_keys, {"gpt2": {}, "some-model-not-in-all-results": {}})

    with pytest.raises(ValueError, match="missing expected key"):
        generate_hf_frontier_leaderboard(str(all_results), str(hf_keys), str(tmp_path / "out"))


def test_generate_our_work_vs_other_approaches_reclassifies_families(tmp_path):
    all_results = tmp_path / "all.json"
    _write_json(all_results, {
        "fanta": _fake_model_entry(8.0),
        "manta": _fake_model_entry(5.0),
        "magnet": _fake_model_entry(4.0),
        "parity_bpe": _fake_model_entry(5.5),
        "flexitokens": _fake_model_entry(23.0),
        "bpe": _fake_model_entry(8.3),  # NOT in either group -- must be excluded
    })

    out_dir = tmp_path / "out"
    rows, families = generate_our_work_vs_other_approaches_leaderboard(str(all_results), str(out_dir))
    assert {r["name"] for r in rows} == {"fanta", "manta", "magnet", "parity_bpe", "flexitokens"}
    by_name = {r["name"]: r for r in rows}
    assert by_name["fanta"]["family"] == "This work"
    for name in ("manta", "magnet", "parity_bpe", "flexitokens"):
        assert by_name[name]["family"] == "Other approaches"
    assert set(families) == {"This work", "Other approaches"}
    # single-column, since only 5 rows -- see gen_spread_leaderboard_tex's own
    # min_rows_for_two_columns default (12)
    tex_path = out_dir / "fig_spread_leaderboard_our_work_vs_other_approaches.tex"
    assert tex_path.read_text().count(r"\begin{tikzpicture}") == 1
