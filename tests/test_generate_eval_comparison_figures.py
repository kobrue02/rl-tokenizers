"""Tests for scripts.generate_eval_comparison_figures: grouped-bar TikZ/
pgfplots figures from combined decoder/encoder eval comparisons. No LaTeX
compiler involved -- these check the .dat tables and the presence/absence
of categories/labels in the generated .tex, matching how
generate_tikz_figures.py's own docstring says to verify (a real compiler,
not this test suite, is what confirms it actually renders)."""

import json
import os

from scripts.generate_eval_comparison_figures import (
    gen_grouped_bar_tex,
    generate_decoder_figures,
    generate_encoder_figures,
)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_gen_grouped_bar_tex_writes_one_dat_file_per_present_label(tmp_path):
    data = {"bpe": {"xnli": 0.42, "xcopa": 0.51}, "fanta": {"xnli": 0.47, "xcopa": 0.55}}
    categories = [("xnli", "XNLI"), ("xcopa", "XCOPA")]

    tex = gen_grouped_bar_tex(data, categories, str(tmp_path), "fig1", ylabel="Score")

    assert "symbolic x coords={xnli,xcopa}" in tex
    assert "{XNLI}, {XCOPA}" in tex
    bpe_dat = _read(tmp_path / "bar_fig1_bpe.dat")
    assert "xnli 0.4200" in bpe_dat
    assert "xcopa 0.5100" in bpe_dat


def test_label_with_no_data_for_this_figure_is_omitted(tmp_path, capsys):
    data = {"bpe": {"xnli": 0.42}, "fanta": {}}
    categories = [("xnli", "XNLI")]

    tex = gen_grouped_bar_tex(data, categories, str(tmp_path), "fig1", ylabel="Score")

    assert "evalColbpe" in tex
    assert "evalColfanta" not in tex
    assert not os.path.exists(tmp_path / "bar_fig1_fanta.dat")
    assert "omitted: ['fanta']" in capsys.readouterr().out


def test_category_no_present_label_has_data_for_is_omitted(tmp_path, capsys):
    data = {"bpe": {"xnli": 0.42}}
    categories = [("xnli", "XNLI"), ("blimp", "BLiMP")]

    tex = gen_grouped_bar_tex(data, categories, str(tmp_path), "fig1", ylabel="Score")

    assert "symbolic x coords={xnli}" in tex
    assert "blimp" not in tex
    assert "omitted: ['blimp']" in capsys.readouterr().out


def test_no_label_has_any_data_returns_none_and_writes_nothing(tmp_path, capsys):
    result = gen_grouped_bar_tex({"bpe": {}}, [("xnli", "XNLI")], str(tmp_path), "fig1", ylabel="Score")

    assert result is None
    assert list(tmp_path.iterdir()) == []
    assert "no label has any data" in capsys.readouterr().out


def test_note_is_appended_to_ylabel(tmp_path):
    tex = gen_grouped_bar_tex(
        {"bpe": {"pppl": 12.3}}, [("pppl", "Pseudoperplexity")], str(tmp_path), "fig1",
        ylabel="Pseudoperplexity", note="lower is better",
    )

    assert "Pseudoperplexity (lower is better)" in tex


def test_generate_decoder_figures_splits_classification_from_flores_mt(tmp_path):
    combined_path = tmp_path / "decoder_comparison.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({
            "bpe": {
                "xnli": {"accuracy": 0.42, "n": 100},
                "xcopa": {"accuracy": 0.51, "n": 100},
                "flores_mt": {"bleu": 3.2, "chrf": 18.5, "n": 50},
            },
        }, f)

    generate_decoder_figures(str(combined_path), str(tmp_path))

    classification_dat = _read(tmp_path / "decoder_classification" / "bar_decoder_classification_bpe.dat")
    assert "xnli 0.4200" in classification_dat
    assert "xcopa 0.5100" in classification_dat
    flores_dat = _read(tmp_path / "decoder_flores_mt" / "bar_decoder_flores_mt_bpe.dat")
    assert "bleu 3.2000" in flores_dat
    assert "chrf 18.5000" in flores_dat


def test_generate_encoder_figures_averages_per_language_suffixed_metrics(tmp_path):
    combined_path = tmp_path / "encoder_comparison.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({
            "bpe": {
                "pppl": {"label": "bpe", "benchmark": "pppl", "result": {"pseudoperplexity": 12.3}},
                "ner": {"label": "bpe", "task": "ner", "result": {"eval_deu_f1": 0.7, "eval_fra_f1": 0.66}},
            },
        }, f)

    generate_encoder_figures(str(combined_path), str(tmp_path))

    classification_dat = _read(tmp_path / "encoder_classification" / "bar_encoder_classification_bpe.dat")
    assert "ner 0.6800" in classification_dat  # mean(0.7, 0.66)
    pppl_dat = _read(tmp_path / "encoder_pseudoperplexity" / "bar_encoder_pseudoperplexity_bpe.dat")
    assert "pppl 12.3000" in pppl_dat
