"""Tests for scripts.generate_morphology_figures: loading/merging
morphology_all.json (+ an optional indigenous_panel companion file),
--exclude, the markdown summary/detailed tables, and the MED-vs-Consistency-F1
scatter figure (well-formedness + own-system labeling)."""

import json

from scripts.generate_morphology_figures import (
    build_detailed_table,
    build_summary_table,
    compute_families,
    gen_morphology_landscape_tex,
    generate,
    load_morphology_rows,
)
from scripts.generate_tikz_figures import _assert_well_formed


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _result(langs):
    """langs: {lang: (med, f1)} -- wraps into the real evaluate.py output shape."""
    return {"morphology": {
        lang: {"med": med, "consistency_f1": f1, "n_words": 100, "source": "morphynet"}
        for lang, (med, f1) in langs.items()
    }}


def test_load_morphology_rows_computes_macro_means_and_sorts_by_med(tmp_path):
    path = tmp_path / "morphology_all.json"
    _write(path, {
        "bpe": _result({"deu_Latn": (4.0, 0.3), "eng_Latn": (2.0, 0.5)}),
        "fanta": _result({"deu_Latn": (1.0, 0.1), "eng_Latn": (1.0, 0.1)}),
        "_failed": {"blt": "gated repo"},
    })
    rows, all_langs = load_morphology_rows(str(path))
    assert all_langs == ["deu_Latn", "eng_Latn"]
    assert [r["name"] for r in rows] == ["fanta", "bpe"]  # fanta's mean_med=1.0 < bpe's 3.0
    assert rows[0]["mean_med"] == 1.0
    assert rows[1]["mean_med"] == 3.0
    assert rows[1]["mean_f1"] == 0.4


def test_load_morphology_rows_excludes_named_systems(tmp_path):
    path = tmp_path / "morphology_all.json"
    _write(path, {"bpe": _result({"eng_Latn": (2.0, 0.5)}), "fanta": _result({"eng_Latn": (1.0, 0.1)})})
    rows, _ = load_morphology_rows(str(path), exclude={"fanta"})
    assert [r["name"] for r in rows] == ["bpe"]


def test_load_morphology_rows_drops_systems_with_zero_scored_languages(tmp_path):
    path = tmp_path / "morphology_all.json"
    _write(path, {"bpe": _result({"eng_Latn": (2.0, 0.5)}), "magnet": {"morphology": {}}})
    rows, _ = load_morphology_rows(str(path))
    assert [r["name"] for r in rows] == ["bpe"]


def test_load_morphology_rows_unions_indigenous_panel_languages_into_same_system(tmp_path):
    """aym/cni/shp come from a SEPARATE indigenous_panel run -- must extend
    the same system's row from --input, not create a second "bpe" row."""
    main_path = tmp_path / "main.json"
    indig_path = tmp_path / "indig.json"
    _write(main_path, {"bpe": _result({"eng_Latn": (2.0, 0.5), "deu_Latn": (4.0, 0.3)})})
    _write(indig_path, {"bpe": _result({"aym": (3.0, 0.2)})})

    rows, all_langs = load_morphology_rows(str(main_path), indigenous_panel_input=str(indig_path))
    assert len(rows) == 1
    assert rows[0]["n_langs"] == 3
    assert set(rows[0]["langs"]) == {"eng_Latn", "deu_Latn", "aym"}
    assert all_langs == ["aym", "deu_Latn", "eng_Latn"]


def test_load_morphology_rows_keeps_system_present_in_only_one_file(tmp_path):
    main_path = tmp_path / "main.json"
    indig_path = tmp_path / "indig.json"
    _write(main_path, {"bpe": _result({"eng_Latn": (2.0, 0.5)})})
    _write(indig_path, {"magnet": _result({"aym": (3.0, 0.2)})})  # magnet has no --input row at all

    rows, _ = load_morphology_rows(str(main_path), indigenous_panel_input=str(indig_path))
    assert {r["name"] for r in rows} == {"bpe", "magnet"}


def test_build_summary_table_reports_family_and_n_langs():
    rows = [{"name": "fanta", "family": "This work", "n_langs": 13, "mean_med": 2.55, "mean_f1": 0.15}]
    header, table_rows = build_summary_table(rows)
    assert header == ["system", "family", "n_langs", "mean_med", "mean_consistency_f1"]
    assert table_rows == [["fanta", "This work", "13", "2.550", "0.150"]]


def test_build_detailed_table_uses_dashes_for_missing_language_coverage():
    rows = [
        {"name": "bpe", "langs": {"eng_Latn": {"med": 2.0, "consistency_f1": 0.5}}},
        {"name": "magnet", "langs": {}},  # magnet scores 0 languages here
    ]
    header, table_rows = build_detailed_table(rows, ["eng_Latn"], "med")
    assert header == ["system", "eng_Latn"]
    assert table_rows == [["bpe", "2.000"], ["magnet", "--"]]


def test_gen_morphology_landscape_tex_is_well_formed_and_labels_own_systems(tmp_path):
    rows, _ = load_morphology_rows(_make_full_fixture(tmp_path))
    families = compute_families(rows)
    tex = gen_morphology_landscape_tex(rows, families, str(tmp_path))
    _assert_well_formed(tex, "fig_morphology_landscape.tex")
    # bpe/fanta are this project's own systems -- always labeled; an
    # external repo is not.
    assert r"{fanta}" in tex
    assert r"{bpe}" in tex
    assert "external-model" not in tex


def _make_full_fixture(tmp_path):
    path = tmp_path / "morphology_all.json"
    _write(path, {
        "bpe": _result({"eng_Latn": (4.0, 0.3), "deu_Latn": (3.0, 0.35)}),
        "fanta": _result({"eng_Latn": (2.0, 0.1), "deu_Latn": (2.5, 0.12)}),
        "external-model": _result({"eng_Latn": (5.0, 0.4), "deu_Latn": (5.5, 0.38)}),
    })
    return str(path)


def test_generate_writes_table_and_figure_end_to_end(tmp_path):
    input_path = _make_full_fixture(tmp_path)
    out_dir = tmp_path / "figs"
    table_output = tmp_path / "morphology_comparison.md"

    rows, all_langs = generate(input_path, str(out_dir), table_output=str(table_output))

    assert len(rows) == 3
    assert (out_dir / "fig_morphology_landscape.tex").exists()
    assert (out_dir / "fig_morphology_landscape_body.tex").exists()
    assert table_output.exists()
    report = table_output.read_text()
    assert "## Summary" in report
    assert "## Detailed: Morphological Edit Distance per language" in report
    assert "## Detailed: Morphological Consistency F1 per language" in report


def test_generate_raises_when_every_system_has_zero_scored_languages(tmp_path):
    path = tmp_path / "morphology_all.json"
    _write(path, {"bpe": {"morphology": {}}})
    try:
        generate(str(path), str(tmp_path / "figs"))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nothing to plot" in str(e)
