"""Tests for scripts/generate_tikz_figures.py's family-membership cleanup,
the landscape duplicate-label bug fix, and the resource-level-trend redesign
(baseline families -- including "Reproduced baselines", the 5 other-researcher
tokenizers this project trained -- aggregate to mean+band; only fanta, this
thesis's own actual contribution, stays individual and highlighted)."""

import csv
import json
import os

import pytest

from scripts import generate_tikz_figures as tikz


def _row(name, family, spread, avg_compression, idx):
    return {
        "name": name,
        "short": tikz.short_name(name),
        "family": family,
        "avg_compression": avg_compression,
        "gini": 0.5,
        "spread": spread,
        "idx": idx,
    }


def test_repo_tokenizer_names_excludes_dropped_systems():
    assert tikz._REPO_TOKENIZER_NAMES == {"bpe", "superbpe", "fanta", "magnet", "manta", "parity_bpe"}
    assert "flexitokens" not in tikz._REPO_TOKENIZER_NAMES
    assert "fairtok" not in tikz._REPO_TOKENIZER_NAMES
    assert tikz.family_of("flexitokens") != "This work"


def test_family_of_distinguishes_fanta_from_reproduced_baselines():
    """Only fanta is this thesis's own contribution -- bpe/superbpe/magnet/
    manta/parity_bpe are other researchers' published methods, reproduced
    here, and must NOT be labeled "This work" (2026-09-16 correction)."""
    assert tikz.family_of("fanta") == "This work"
    for name in ("bpe", "superbpe", "magnet", "manta", "parity_bpe"):
        assert tikz.family_of(name) == "Reproduced baselines"


def test_landscape_dedup_no_duplicate_labels(tmp_path):
    # "outlier" is deliberately both the max-spread AND max-compression row --
    # the exact scenario that produced a duplicated, overlapping label before
    # the dedup fix.
    rows = [
        _row("bpe", "This work", spread=5.0, avg_compression=3.0, idx=0),
        _row("outlier", "Other", spread=25.0, avg_compression=27.0, idx=1),
        _row("magnet", "This work", spread=4.0, avg_compression=6.0, idx=2),
    ]
    families = tikz.compute_families(rows)
    tex = tikz.gen_landscape_tex(rows, families, str(tmp_path))
    # Exactly one \node per labeled point, even though "outlier" qualifies
    # for two of the three superlative slots (worst_spread and best_compression).
    assert tex.count(r"\node[font=\scriptsize") == 2
    assert tex.count("outlier") == 1


def test_resource_level_aggregates_baselines_keeps_fanta_individual(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tikz, "load_resource_levels",
        lambda codes: ({"en": 5, "lang_a": 1, "lang_b": 1}, []),
    )
    rows = [
        _row("fanta", "This work", spread=2.5, avg_compression=3.2, idx=0),
        _row("bpe", "Reproduced baselines", spread=2.0, avg_compression=3.0, idx=1),
        _row("modelA", "Chinese labs", spread=10.0, avg_compression=4.0, idx=2),
        _row("modelB", "Chinese labs", spread=12.0, avg_compression=4.5, idx=3),
    ]
    models = {
        "fanta": {"token_parity": {"en": 1.0, "lang_a": 1.3, "lang_b": 1.1}},
        "bpe": {"token_parity": {"en": 1.0, "lang_a": 1.1, "lang_b": 1.2}},
        "modelA": {"token_parity": {"en": 1.0, "lang_a": 2.0, "lang_b": 4.0}},
        "modelB": {"token_parity": {"en": 1.0, "lang_a": 3.0, "lang_b": 5.0}},
    }
    families = tikz.compute_families(rows)
    out_dir = str(tmp_path)
    tex, counts, unresolved = tikz.gen_resource_level_tex(rows, models, families, out_dir)

    # Only fanta (idx 0, "This work") gets an individual .dat file.
    assert os.path.exists(os.path.join(out_dir, "resourcelevel_0.dat"))
    assert not os.path.exists(os.path.join(out_dir, "resourcelevel_1.dat"))
    assert not os.path.exists(os.path.join(out_dir, "resourcelevel_2.dat"))
    assert not os.path.exists(os.path.join(out_dir, "resourcelevel_3.dat"))
    # bpe -- a reproduced baseline, NOT this thesis's contribution -- aggregates
    # into its own family file exactly like any external baseline family does.
    repro_path = os.path.join(out_dir, f"resourcelevel_family_{tikz.fam_key('Reproduced baselines')}.dat")
    assert os.path.exists(repro_path)
    china_path = os.path.join(out_dir, f"resourcelevel_family_{tikz.fam_key('Chinese labs')}.dat")
    assert os.path.exists(china_path)

    with open(china_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # level 1 (lang_a + lang_b): modelA mean=(2.0+4.0)/2=3.0, modelB mean=(3.0+5.0)/2=4.0
    # -> family mean 3.5, lo 3.0, hi 4.0
    assert "1 3.5000 3.0000 4.0000" in lines

    # Figure references fillbetween (the shaded bands) for both baseline
    # families, one highlighted star-marker addplot for fanta alone, and no
    # individual addplot for bpe (it only appears via the aggregate family line).
    assert r"\usepgfplotslibrary{fillbetween}" in tex
    assert "fill between" in tex
    assert tex.count(r"\addlegendentry{fanta}") == 1
    assert tex.count(r"\addlegendentry{bpe}") == 0
    assert tex.count(r"\addlegendentry{Reproduced baselines}") == 1
    assert tex.count(r"\addlegendentry{Chinese labs}") == 1
    assert "mark=star" in tex


def test_resource_level_single_member_family_degenerate_band(tmp_path, monkeypatch):
    monkeypatch.setattr(tikz, "load_resource_levels", lambda codes: ({"lang_a": 2}, []))
    rows = [_row("solo-model", "Anthropic", spread=3.0, avg_compression=2.0, idx=0)]
    models = {"solo-model": {"token_parity": {"lang_a": 1.5}}}
    families = tikz.compute_families(rows)
    out_dir = str(tmp_path)
    tikz.gen_resource_level_tex(rows, models, families, out_dir)
    fam_path = os.path.join(out_dir, f"resourcelevel_family_{tikz.fam_key('Anthropic')}.dat")
    with open(fam_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert "2 1.5000 1.5000 1.5000" in lines


def test_gen_tokenizer_summary_table_tex(tmp_path):
    rows = [
        _row("bpe", "This work", spread=2.0, avg_compression=3.0, idx=0),
        _row("modelA", "Chinese labs", spread=10.0, avg_compression=4.0, idx=1),
    ]
    models = {
        "bpe": {"fertility": {"lang_a": 1.0, "lang_b": 1.4}},  # mean 1.2
        "modelA": {"fertility": {"lang_a": 1.6, "lang_b": 2.0}},  # mean 1.8
    }
    tex = tikz.gen_tokenizer_summary_table_tex(rows, models, str(tmp_path))
    assert tex.count(r"\begin{tikzpicture}") == 0
    assert "bpe & This work & 3.00 & 1.20 & 0.500 & 2.00" in tex
    assert "modelA & Chinese labs & 4.00 & 1.80 & 0.500 & 10.00" in tex


def test_gen_resource_level_table_tex_marks_missing_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tikz, "load_resource_levels", lambda codes: ({"lang_a": 0, "lang_b": 3}, []),
    )
    rows = [_row("bpe", "This work", spread=2.0, avg_compression=3.0, idx=0)]
    # bpe only has data at level 0, not level 3 -- table should show "--" there.
    models = {"bpe": {"token_parity": {"lang_a": 1.5}}}
    tex = tikz.gen_resource_level_table_tex(rows, models, str(tmp_path))
    assert "bpe & This work & 1.50 & --" in tex


def test_generate_end_to_end_below_two_column_threshold(tmp_path):
    """generate()'s own well-formedness check for the leaderboard figure was
    hardcoded to expect the 2-column layout unconditionally -- harmless in
    practice (every real run has well over
    MIN_ROWS_FOR_TWO_COLUMN_LEADERBOARD rows) but wrong below that threshold,
    which this small fixture deliberately exercises."""
    data = {
        "fanta": {
            "avg_compression": 3.2, "fertility": {"en": 1.1}, "gini": 0.3,
            "token_parity": {"en": 1.0, "de": 1.2}, "token_parity_spread": 2.0,
        },
        "bpe": {
            "avg_compression": 3.0, "fertility": {"en": 1.3}, "gini": 0.35,
            "token_parity": {"en": 1.0, "de": 1.5}, "token_parity_spread": 3.0,
        },
    }
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps(data))
    rows, _families, _top_langs = tikz.generate(str(input_path), str(tmp_path / "out"))
    by_name = {r["name"]: r["family"] for r in rows}
    assert by_name["fanta"] == "This work"
    assert by_name["bpe"] == "Reproduced baselines"


def test_load_failed_reads_the_failed_key(tmp_path):
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps({
        "bpe": {"token_parity": {"en": 1.0}},
        "_failed": {"some/gated-repo": "GatedRepoError: access denied"},
    }))
    assert tikz.load_failed(str(input_path)) == {"some/gated-repo": "GatedRepoError: access denied"}


def test_load_failed_absent_key_returns_empty(tmp_path):
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps({"bpe": {"token_parity": {"en": 1.0}}}))
    assert tikz.load_failed(str(input_path)) == {}


def test_gen_coverage_table_tex(tmp_path):
    rows = [
        _row("fanta", "This work", spread=2.0, avg_compression=3.0, idx=0),
        _row("bpe", "Reproduced baselines", spread=2.5, avg_compression=3.2, idx=1),
    ]
    models = {
        "fanta": {"token_parity": {"en": 1.0, "de": 1.2, "fr": 1.1}},
        "bpe": {"token_parity": {"en": 1.0, "de": 1.5}},  # missing "fr" -- partial coverage
    }
    failed = {"some/gated-repo": "GatedRepoError: access denied"}
    tex = tikz.gen_coverage_table_tex(rows, models, failed, str(tmp_path))
    assert tex.count(r"\begin{tikzpicture}") == 0
    assert "fanta & This work & 3/3 (100\\%)" in tex
    assert "bpe & Reproduced baselines & 2/3 (67\\%)" in tex
    assert "FAILED (GatedRepoError: access denied)" in tex


def test_gen_coverage_table_tex_truncates_long_error_messages(tmp_path):
    rows = [_row("fanta", "This work", spread=2.0, avg_compression=3.0, idx=0)]
    models = {"fanta": {"token_parity": {"en": 1.0}}}
    failed = {"some/repo": "X" * 200}
    tex = tikz.gen_coverage_table_tex(rows, models, failed, str(tmp_path))
    # A table cell, not a log -- must not dump the full 200-char message.
    assert "X" * 200 not in tex
    assert "..." in tex


def test_write_full_csv(tmp_path):
    rows = [
        _row("fanta", "This work", spread=2.0, avg_compression=3.0, idx=0),
        _row("bpe", "Reproduced baselines", spread=2.5, avg_compression=3.2, idx=1),
    ]
    models = {
        "fanta": {
            "token_parity": {"en": 1.0, "de": 1.2},
            "token_parity_gm": {"en": 0.9, "de": 1.0},
            "fertility": {"en": 1.1, "de": 1.3},
            "per_lang_compression": {"en": 3.0, "de": 2.8},
            "renyi": {"en": 0.6, "de": 0.65},
        },
        "bpe": {"token_parity": {"en": 1.0}},  # missing every other per-language field
    }
    out_path = tmp_path / "full.csv"
    tikz.write_full_csv(rows, models, str(out_path))

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 3  # fanta x2 languages + bpe x1 language
    by_key = {(r["tokenizer"], r["language"]): r for r in reader}
    assert by_key[("fanta", "de")]["token_parity"] == "1.2"
    assert by_key[("fanta", "de")]["fertility"] == "1.3"
    assert by_key[("bpe", "en")]["token_parity"] == "1.0"
    assert by_key[("bpe", "en")]["fertility"] == ""  # missing field -> blank, not a crash


def test_generate_writes_csv_when_requested(tmp_path):
    data = {"fanta": {
        "avg_compression": 3.2, "fertility": {"en": 1.1}, "gini": 0.3,
        "token_parity": {"en": 1.0, "de": 1.2}, "token_parity_spread": 2.0,
    }}
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps(data))
    csv_path = tmp_path / "full.csv"
    tikz.generate(str(input_path), str(tmp_path / "out"), csv_out=str(csv_path))
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["language"] for r in rows} == {"en", "de"}
