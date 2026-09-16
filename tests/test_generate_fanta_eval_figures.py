"""Tests for scripts.generate_fanta_eval_figures's resource-level improvement
plot (fanta's mean token_parity improvement over each reproduced baseline,
per Joshi resource level) -- a delta view of Ch.~tokentax's own
fig:resource-level-trend, for Ch.~fantaeval."""

import json
import os

import pytest

from scripts.generate_fanta_eval_figures import generate_resource_level_improvement


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _model_entry(token_parity):
    return {
        "avg_compression": 2.0, "gini": 0.05, "token_parity_spread": 3.0,
        "token_parity": token_parity,
    }


def test_generate_resource_level_improvement_computes_baseline_minus_fanta(tmp_path, monkeypatch):
    import scripts.generate_fanta_eval_figures as mod

    level_1_mean_by_name = {
        "fanta": 1.0, "bpe": 1.5, "superbpe": 2.0, "magnet": 1.2, "manta": 1.1,
        "parity_bpe": 0.9,  # worse than fanta here -- negative improvement
    }

    def _fake_resource_levels_and_means(rows, models):
        row_level_means = {r["idx"]: {1: level_1_mean_by_name[r["name"]]} for r in rows}
        return {"en": 5, "lang_a": 1, "lang_b": 1}, [], [1], {1: 2}, row_level_means

    monkeypatch.setattr(mod, "_resource_levels_and_means", _fake_resource_levels_and_means)

    all_results = tmp_path / "all.json"
    _write_json(all_results, {
        name: _model_entry({"en": 1.0, "lang_a": 1.0, "lang_b": 1.0})
        for name in ("fanta", "bpe", "superbpe", "magnet", "manta", "parity_bpe")
    })
    out_dir = tmp_path / "out"
    tex = generate_resource_level_improvement(str(all_results), str(out_dir))

    assert tex.count(r"\begin{tikzpicture}") == 1
    with open(out_dir / "improvement_bpe.dat", encoding="utf-8") as f:
        assert "1 0.5000" in f.read()  # 1.5 - 1.0
    with open(out_dir / "improvement_parity_bpe.dat", encoding="utf-8") as f:
        assert "1 -0.1000" in f.read()  # 0.9 - 1.0 -- fanta worse than this baseline here
    assert tex.count(r"\addlegendentry{") == 5  # one per reproduced baseline


def test_generate_resource_level_improvement_raises_on_missing_tokenizer(tmp_path):
    all_results = tmp_path / "all.json"
    _write_json(all_results, {"fanta": _model_entry({"en": 1.0})})  # missing all 5 baselines
    with pytest.raises(ValueError, match="missing expected tokenizer"):
        generate_resource_level_improvement(str(all_results), str(tmp_path / "out"))
