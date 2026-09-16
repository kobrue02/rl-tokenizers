"""Tests for scripts/generate_tikz_figures.py's family-membership cleanup,
the landscape duplicate-label bug fix, and the resource-level-trend redesign
(baseline families aggregate to mean+band, "This work" stays individual)."""

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
    assert tikz.family_of("bpe") == "This work"


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


def test_resource_level_aggregates_baselines_keeps_own_individual(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tikz, "load_resource_levels",
        lambda codes: ({"en": 5, "lang_a": 1, "lang_b": 1}, []),
    )
    rows = [
        _row("bpe", "This work", spread=2.0, avg_compression=3.0, idx=0),
        _row("fanta", "This work", spread=2.5, avg_compression=3.2, idx=1),
        _row("modelA", "Chinese labs", spread=10.0, avg_compression=4.0, idx=2),
        _row("modelB", "Chinese labs", spread=12.0, avg_compression=4.5, idx=3),
    ]
    models = {
        "bpe": {"token_parity": {"en": 1.0, "lang_a": 1.1, "lang_b": 1.2}},
        "fanta": {"token_parity": {"en": 1.0, "lang_a": 1.3, "lang_b": 1.1}},
        "modelA": {"token_parity": {"en": 1.0, "lang_a": 2.0, "lang_b": 4.0}},
        "modelB": {"token_parity": {"en": 1.0, "lang_a": 3.0, "lang_b": 5.0}},
    }
    families = tikz.compute_families(rows)
    out_dir = str(tmp_path)
    tex, counts, unresolved = tikz.gen_resource_level_tex(rows, models, families, out_dir)

    # Own-system rows still get their own individual .dat file.
    assert os.path.exists(os.path.join(out_dir, "resourcelevel_0.dat"))
    assert os.path.exists(os.path.join(out_dir, "resourcelevel_1.dat"))
    # Baseline family aggregates into one file, not one per tokenizer.
    fam_path = os.path.join(out_dir, f"resourcelevel_family_{tikz.fam_key('Chinese labs')}.dat")
    assert os.path.exists(fam_path)
    assert not os.path.exists(os.path.join(out_dir, "resourcelevel_2.dat"))
    assert not os.path.exists(os.path.join(out_dir, "resourcelevel_3.dat"))

    with open(fam_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # level 1 (lang_a + lang_b): modelA mean=(2.0+4.0)/2=3.0, modelB mean=(3.0+5.0)/2=4.0
    # -> family mean 3.5, lo 3.0, hi 4.0
    assert "1 3.5000 3.0000 4.0000" in lines

    # Figure references fillbetween (the shaded band) and one addplot per
    # own-system row with its own marker shape + legend entry, but no
    # per-row addplot for the aggregated baseline family.
    assert r"\usepgfplotslibrary{fillbetween}" in tex
    assert "fill between" in tex
    assert tex.count(r"\addlegendentry{bpe}") == 1
    assert tex.count(r"\addlegendentry{fanta}") == 1
    assert tex.count(r"\addlegendentry{Chinese labs}") == 1
    assert "mark=square*" not in tex  # superbpe's mark, unused here
    assert "mark=*" in tex  # bpe's own mark


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
