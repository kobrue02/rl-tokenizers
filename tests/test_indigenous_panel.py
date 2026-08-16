"""Tests for the indigenous_panel data source (parsing/writing/reading logic)
against small synthetic fixtures, rather than the real network-dependent
sources (~202MB for Nunavut Hansard) that prepare_indigenous_panel actually
talks to."""

import io
import json
import os
import tarfile

import pytest

from common.data import corpora
from common.data.indigenous_panel import NRC_HANSARD_ARCHIVE_ROOT
from common.data.prepare_indigenous_panel import _extract_nrc_hansard_test_split, _write_pairs_jsonl
from common.eval.cross_tokenizer import evaluate_on_indigenous_panel, run_eval_cli


def _make_synthetic_hansard_tgz(path, en_lines, iu_lines):
    with tarfile.open(path, "w:gz") as tar:
        for suffix, lines in ((".en", en_lines), (".iu", iu_lines)):
            data = ("\n".join(lines) + "\n").encode("utf-8")
            info = tarfile.TarInfo(name=f"{NRC_HANSARD_ARCHIVE_ROOT}/split/test{suffix}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_nrc_hansard_test_split_zips_lines_correctly(tmp_path):
    tgz_path = tmp_path / "fake_hansard.tgz"
    _make_synthetic_hansard_tgz(tgz_path, ["Nunavut Canada", "Good afternoon"], ["ᐃᐊᑐᛖ", "ᒃᖃ"])
    rows = _extract_nrc_hansard_test_split(str(tgz_path))
    assert rows == [
        {"en": "Nunavut Canada", "iu": "ᐃᐊᑐᛖ"},
        {"en": "Good afternoon", "iu": "ᒃᖃ"},
    ]


def test_extract_nrc_hansard_test_split_drops_blank_lines(tmp_path):
    tgz_path = tmp_path / "fake_hansard.tgz"
    _make_synthetic_hansard_tgz(tgz_path, ["one", "", "three"], ["ONE", "", "THREE"])
    rows = _extract_nrc_hansard_test_split(str(tgz_path))
    assert rows == [{"en": "one", "iu": "ONE"}, {"en": "three", "iu": "THREE"}]


def test_extract_nrc_hansard_test_split_raises_on_length_mismatch(tmp_path):
    tgz_path = tmp_path / "fake_hansard.tgz"
    _make_synthetic_hansard_tgz(tgz_path, ["one", "two"], ["ONE"])
    with pytest.raises(ValueError, match="line count mismatch"):
        _extract_nrc_hansard_test_split(str(tgz_path))


def test_corpora_stream_indigenous_panel_single_pair(tmp_path):
    rows = [{"crk": "kiya", "en": "you"}, {"crk": "niya", "en": "me"}]
    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", rows)
    result = list(corpora._stream_indigenous_panel_single("crk-en", output_dir=str(tmp_path)))
    assert result == rows


def test_corpora_list_indigenous_panel_pairs(tmp_path):
    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "a", "en": "b"}])
    _write_pairs_jsonl(tmp_path / "nah-es.jsonl", [{"nah": "c", "es": "d"}])
    (tmp_path / "metadata.json").write_text("{}")  # non-.jsonl file, must be ignored
    pairs = corpora.list_indigenous_panel_pairs(output_dir=str(tmp_path))
    assert pairs == ["crk-en", "nah-es"]


def test_corpora_list_indigenous_panel_pairs_missing_dir_returns_empty(tmp_path):
    assert corpora.list_indigenous_panel_pairs(output_dir=str(tmp_path / "does_not_exist")) == []


def test_corpora_stream_indigenous_panel_single_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="one-time local prep"):
        list(corpora._stream_indigenous_panel_single("crk-en", output_dir=str(tmp_path)))


def test_stream_groups_indigenous_panel_round_robins_multiple_pairs(tmp_path):
    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "a1", "en": "b1"}, {"crk": "a2", "en": "b2"}])
    _write_pairs_jsonl(tmp_path / "nah-es.jsonl", [{"nah": "c1", "es": "d1"}])
    orig_dir = corpora.INDIGENOUS_PANEL_LOCAL_DIR
    corpora.INDIGENOUS_PANEL_LOCAL_DIR = str(tmp_path)
    try:
        groups = list(corpora.stream_groups("indigenous_panel", config="all"))
    finally:
        corpora.INDIGENOUS_PANEL_LOCAL_DIR = orig_dir
    assert {"crk": "a1", "en": "b1"} in groups
    assert {"nah": "c1", "es": "d1"} in groups
    assert len(groups) == 3


def _one_token_per_byte(raw):
    return [bytes([b]) for b in raw]


def test_evaluate_on_indigenous_panel_separates_anchors():
    induce_fn_by_lang = {
        lang: _one_token_per_byte for lang in ("crk", "en", "iu", "nah", "es")
    }
    eval_groups = [
        {"crk": "ab", "en": "abcd"},  # crk needs half as many tokens as en
        {"iu": "abcdefgh", "en": "abcd"},  # iu needs twice as many as en
        {"nah": "abcdef", "es": "abc"},  # nah needs twice as many as es
    ]
    results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)

    combined = results["combined"]
    assert "token_parity" not in combined
    assert "token_parity_anchor" not in combined
    assert "token_parity_gm" not in combined
    assert "token_parity_spread" not in combined
    assert set(combined["fertility"]) == {"crk", "en", "iu", "nah", "es"}

    en_scope = results["token_parity_by_anchor"]["en"]
    assert en_scope["token_parity"]["crk"] == pytest.approx(0.5)
    assert en_scope["token_parity"]["iu"] == pytest.approx(2.0)
    assert "nah" not in en_scope["token_parity"]  # never paired with "en"

    es_scope = results["token_parity_by_anchor"]["es"]
    assert es_scope["token_parity"]["nah"] == pytest.approx(2.0)
    assert "crk" not in es_scope["token_parity"]  # never paired with "es"

    # morphology_spread needs no anchor at all -- max/min fertility across
    # every language in the whole panel, English/Spanish included.
    assert results["morphology_spread"]["fertility_spread"] >= 1.0


def test_evaluate_on_indigenous_panel_raises_on_unrecognized_anchor():
    induce_fn_by_lang = {"crk": _one_token_per_byte, "de": _one_token_per_byte}
    with pytest.raises(ValueError, match="known anchor languages"):
        evaluate_on_indigenous_panel(induce_fn_by_lang, [{"crk": "ab", "de": "cd"}])


def test_hf_frontier_evaluate_end_to_end_on_indigenous_panel(tmp_path, monkeypatch):
    """Exercises the full --eval-data-source indigenous_panel path through
    systems.tokenization.hf_frontier.evaluate.main (CLI parsing, group loading, the
    evaluate_on_indigenous_panel branch, token_freq-stripped JSON output)
    against a tiny local fixture panel with a real gpt2 tokenizer (small/fast
    network load, not offline)."""
    from systems.tokenization.hf_frontier.evaluate import main

    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "namoya", "en": "no"}])
    _write_pairs_jsonl(tmp_path / "nah-es.jsonl", [{"nah": "amo", "es": "no"}])
    monkeypatch.setattr(corpora, "INDIGENOUS_PANEL_LOCAL_DIR", str(tmp_path))

    out_path = tmp_path / "results.json"
    all_results = main([
        "--hf-repo-id", "gpt2",
        "--eval-data-source", "indigenous_panel",
        "--output", str(out_path),
    ])

    result = all_results["gpt2"]
    assert set(result) == {"combined", "token_parity_by_anchor", "morphology_spread"}
    assert "token_freq" not in result["combined"]
    assert set(result["token_parity_by_anchor"]) == {"en", "es"}
    for anchor_results in result["token_parity_by_anchor"].values():
        assert "token_freq" not in anchor_results

    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded == all_results  # confirms the JSON actually round-trips cleanly


def test_generate_indigenous_panel_figures(tmp_path):
    """generate_indigenous_panel_figures against a hand-built results.json
    matching evaluate_on_indigenous_panel's shape -- confirms one
    parity_vs_<anchor>/ subdirectory per anchor, each a well-formed grouped
    bar chart (one .dat per tokenizer family, mean token_parity per family
    per language). This grouped-bar design replaced an earlier
    per-language-subplot layout that overflowed the page at 10 languages."""
    from scripts.generate_tikz_figures import generate_indigenous_panel_figures

    def fake_model_result(fertility_spread):
        return {
            "combined": {
                "avg_compression": 2.5, "gini": 0.1,
                "fertility": {}, "per_lang_compression": {}, "renyi": {},
            },
            "token_parity_by_anchor": {
                "en": {"token_parity": {"crk": 1.2, "iu": 0.9}, "token_parity_gm": {"crk": 1.1, "iu": 0.95}},
                "es": {"token_parity": {c: 1.0 for c in
                       ("nah", "hch", "oto", "gn", "bzd", "quy", "aym", "tar", "shp", "cni")},
                       "token_parity_gm": {}},
            },
            "morphology_spread": {"fertility_spread": fertility_spread, "compression_spread": 1.5},
        }

    fake_results = {
        "gpt2": fake_model_result(5.0),
        "bert-base-cased": fake_model_result(3.0),
    }
    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(fake_results, f)

    out_dir = tmp_path / "figures"
    rows, families, written = generate_indigenous_panel_figures(str(results_path), str(out_dir))

    assert set(written) == {"en", "es"}
    assert written["en"] == ["crk", "iu"]
    assert len(written["es"]) == 10
    for anchor in ("en", "es"):
        anchor_dir = out_dir / f"parity_vs_{anchor}"
        tex_path = anchor_dir / f"fig_indigenous_panel_parity_vs_{anchor}.tex"
        assert tex_path.exists()
        assert (anchor_dir / f"fig_indigenous_panel_parity_vs_{anchor}_body.tex").exists()
        # one .dat per tokenizer family, mean-aggregated (not per-model)
        for fam in families:
            assert (anchor_dir / f"bar_indigenous_panel_parity_vs_{anchor}_{fam.replace('/', '_').replace(' ', '_')}.dat").exists()


def test_gen_indigenous_panel_parity_bars_omits_family_with_no_data(tmp_path):
    """Regression test: a family with ZERO data points for an anchor group must
    have its \\addplot/\\addlegendentry pair skipped, not emitted against an
    empty table. pgfplots silently drops an empty addplot from its own legend
    numbering, shifting every LATER \\addlegendentry to mislabel the next plot
    (confirmed on a real Overleaf render -- "Anthropic" ended up labeled
    "Other"). Locks in the fix: the empty family's .dat/TeX entries must not
    exist."""
    from scripts.generate_tikz_figures import gen_indigenous_panel_parity_bars_tex

    rows = [
        {"name": "gpt2", "family": "OpenAI/tiktoken"},
        {"name": "claude-opus-5", "family": "Anthropic"},
    ]
    models = {
        "gpt2": {"token_parity_by_anchor": {"en": {"token_parity": {"crk": 1.2, "iu": 0.9}}}},
        "claude-opus-5": {"token_parity_by_anchor": {"en": {"token_parity": {}}}},  # no data yet
    }
    families = ["OpenAI/tiktoken", "Anthropic"]

    tex = gen_indigenous_panel_parity_bars_tex(
        rows, models, families, ["crk", "iu"], "en", "test_fig", str(tmp_path)
    )

    assert r"\addlegendentry{Anthropic}" not in tex
    assert "fill=anthropicCol" not in tex  # no addplot referencing it either
    assert r"\addlegendentry{OpenAI/tiktoken}" in tex
    assert not os.path.exists(tmp_path / "bar_test_fig_Anthropic.dat")
    assert os.path.exists(tmp_path / "bar_test_fig_OpenAI_tiktoken.dat")


def test_run_eval_cli_indigenous_panel_branch(tmp_path, monkeypatch):
    """run_eval_cli is the shared main() body for all seven systems' own
    evaluate.py, so extending it once here covers all of them. Uses a fake
    load_model/build_induce_fn_by_lang since the branch under test doesn't
    depend on which real system is calling in."""
    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "namoya", "en": "no way"}])
    _write_pairs_jsonl(tmp_path / "nah-es.jsonl", [{"nah": "amo", "es": "no"}])
    monkeypatch.setattr(corpora, "INDIGENOUS_PANEL_LOCAL_DIR", str(tmp_path))

    def fake_load_model(checkpoint, device):
        return object()

    def fake_build_induce_fn_by_lang(model, sequences_by_lang, args):
        return {lang: _one_token_per_byte for lang in sequences_by_lang}

    results = run_eval_cli(
        ["--checkpoint", "unused", "--eval-data-source", "indigenous_panel"],
        "fake_system",
        fake_load_model,
        fake_build_induce_fn_by_lang,
    )
    assert set(results) == {"combined", "token_parity_by_anchor", "morphology_spread"}
    assert set(results["token_parity_by_anchor"]) == {"en", "es"}


def test_run_eval_cli_bouquet_branch_unaffected(monkeypatch):
    """Confirms extending run_eval_cli for indigenous_panel didn't change its
    default (bouquet) behavior; uses a synthetic_groups override to stay
    offline/fast rather than a real BOUQuET fetch."""
    fake_groups = [{"eng": "hello world", "deu": "hallo welt"}]

    def fake_load_model(checkpoint, device):
        return object()

    def fake_build_induce_fn_by_lang(model, sequences_by_lang, args):
        return {lang: _one_token_per_byte for lang in sequences_by_lang}

    results = run_eval_cli(
        ["--checkpoint", "unused", "--eval-data-source", "synthetic"],
        "fake_system",
        fake_load_model,
        fake_build_induce_fn_by_lang,
        synthetic_groups=fake_groups,
    )
    assert set(results) == {
        "token_freq", "renyi", "gini", "per_lang_compression", "avg_compression",
        "fertility", "token_parity", "token_parity_anchor", "token_parity_gm", "token_parity_spread",
    }


def test_run_eval_cli_output_writes_combinable_json(tmp_path):
    """--output writes {result_key: results} JSON with token_freq stripped
    (bytes keys aren't JSON-serializable), matching the shape
    scripts/combine_eval_results.py expects. result_key defaults to system_label;
    --result-key overrides it to keep differently-configured runs of the
    same system as distinct entries."""
    fake_groups = [{"eng": "hello world", "deu": "hallo welt"}]

    def fake_load_model(checkpoint, device):
        return object()

    def fake_build_induce_fn_by_lang(model, sequences_by_lang, args):
        return {lang: _one_token_per_byte for lang in sequences_by_lang}

    out_path = tmp_path / "results.json"
    run_eval_cli(
        ["--checkpoint", "unused", "--eval-data-source", "synthetic", "--output", str(out_path)],
        "fanta",
        fake_load_model,
        fake_build_induce_fn_by_lang,
        synthetic_groups=fake_groups,
    )
    with open(out_path) as f:
        payload = json.load(f)
    assert set(payload) == {"fanta"}  # defaults to system_label
    assert "token_freq" not in payload["fanta"]
    assert "avg_compression" in payload["fanta"]

    out_path2 = tmp_path / "results2.json"
    run_eval_cli(
        [
            "--checkpoint", "unused", "--eval-data-source", "synthetic",
            "--output", str(out_path2), "--result-key", "fanta_variant_b",
        ],
        "fanta",
        fake_load_model,
        fake_build_induce_fn_by_lang,
        synthetic_groups=fake_groups,
    )
    with open(out_path2) as f:
        payload2 = json.load(f)
    assert set(payload2) == {"fanta_variant_b"}


def test_run_eval_cli_output_indigenous_panel_strips_nested_token_freq(tmp_path, monkeypatch):
    """Same --output guarantee, for the nested indigenous_panel shape
    (token_freq appears both inside "combined" and inside each anchor's
    entry in "token_parity_by_anchor")."""
    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "namoya", "en": "no way"}])
    monkeypatch.setattr(corpora, "INDIGENOUS_PANEL_LOCAL_DIR", str(tmp_path))

    def fake_load_model(checkpoint, device):
        return object()

    def fake_build_induce_fn_by_lang(model, sequences_by_lang, args):
        return {lang: _one_token_per_byte for lang in sequences_by_lang}

    out_path = tmp_path / "results.json"
    run_eval_cli(
        ["--checkpoint", "unused", "--eval-data-source", "indigenous_panel", "--output", str(out_path)],
        "manta",
        fake_load_model,
        fake_build_induce_fn_by_lang,
    )
    with open(out_path) as f:
        payload = json.load(f)
    assert set(payload) == {"manta"}
    result = payload["manta"]
    assert "token_freq" not in result["combined"]
    for anchor_results in result["token_parity_by_anchor"].values():
        assert "token_freq" not in anchor_results


def test_run_eval_cli_accepts_yaml_config(tmp_path):
    """run_eval_cli must accept -c/--config like every other evaluate.py/train.py
    in this repo (via parse_args_with_config) -- an earlier version used plain
    argparse.parse_args and silently missed this. --checkpoint is required, and
    must be satisfiable from the YAML file alone, not just the CLI."""
    config_path = tmp_path / "eval_config.yml"
    config_path.write_text("checkpoint: fake_checkpoint_path\neval_data_source: synthetic\n")

    def fake_load_model(checkpoint, device):
        assert checkpoint == "fake_checkpoint_path"
        return object()

    def fake_build_induce_fn_by_lang(model, sequences_by_lang, args):
        return {lang: _one_token_per_byte for lang in sequences_by_lang}

    fake_groups = [{"eng": "hello world", "deu": "hallo welt"}]
    results = run_eval_cli(
        ["-c", str(config_path)],
        "fake_system",
        fake_load_model,
        fake_build_induce_fn_by_lang,
        synthetic_groups=fake_groups,
    )
    assert results["avg_compression"] > 0


def _fake_word_count_fn(text):
    return len(text.split())


def test_evaluate_claude_on_indigenous_panel_separates_anchors():
    from systems.tokenization.claude_tokenizer.evaluate import evaluate_claude_on_indigenous_panel

    eval_groups = [
        {"crk": "ab cd", "en": "a b c d e f g h"},  # crk needs half as many "tokens" as en
        {"iu": "ab cd ef gh ij kl mn op", "en": "a b c d"},  # iu needs twice as many as en
        {"nah": "ab cd ef", "es": "a b"},  # nah needs 1.5x as many as es
    ]
    results = evaluate_claude_on_indigenous_panel(eval_groups, _fake_word_count_fn, max_workers=2, progress_every=0)

    assert set(results["token_parity_by_anchor"]) == {"en", "es"}
    en_scope = results["token_parity_by_anchor"]["en"]
    assert en_scope["token_parity"]["crk"] == pytest.approx(0.25)
    assert en_scope["token_parity"]["iu"] == pytest.approx(2.0)
    es_scope = results["token_parity_by_anchor"]["es"]
    assert es_scope["token_parity"]["nah"] == pytest.approx(1.5)

    combined = results["combined"]
    assert combined["renyi"] == {}
    assert combined["gini"] is None
    assert set(combined["fertility"]) == {"crk", "en", "iu", "nah", "es"}
    assert results["num_total_calls"] == sum(len(g) for g in eval_groups)
    assert results["num_failed_calls"] == 0


def test_evaluate_claude_on_indigenous_panel_checkpoint_resume(tmp_path):
    """Each anchor gets its own checkpoint file -- confirms a pre-seeded
    checkpoint for ONE anchor is respected without touching the other."""
    from systems.tokenization.claude_tokenizer.evaluate import evaluate_claude_on_indigenous_panel

    eval_groups = [{"crk": "ab cd", "en": "a b c d"}, {"nah": "ab", "es": "a b c"}]
    ckpt_base = str(tmp_path / "claude-model")

    queried = []

    def spy_count(text):
        queried.append(text)
        return _fake_word_count_fn(text)

    first = evaluate_claude_on_indigenous_panel(
        eval_groups, spy_count, max_workers=2, progress_every=0, checkpoint_path=ckpt_base
    )
    assert first["num_skipped_via_checkpoint"] == 0
    assert os.path.exists(f"{ckpt_base}.en.jsonl")
    assert os.path.exists(f"{ckpt_base}.es.jsonl")

    queried.clear()
    resumed = evaluate_claude_on_indigenous_panel(
        eval_groups, spy_count, max_workers=2, progress_every=0, checkpoint_path=ckpt_base
    )
    assert queried == [], "every (group, lang) pair was already checkpointed -- nothing should be re-queried"
    assert resumed["num_skipped_via_checkpoint"] == first["num_total_calls"]
    assert resumed["combined"]["fertility"] == first["combined"]["fertility"]


def test_claude_evaluate_main_end_to_end_on_indigenous_panel(tmp_path, monkeypatch):
    """Exercises the full --eval-data-source indigenous_panel path through
    systems.tokenization.claude_tokenizer.evaluate.main, with ClaudeTokenCounter
    monkeypatched to a fake so this stays fast, offline, and credential-free."""
    import systems.tokenization.claude_tokenizer.evaluate as claude_evaluate

    _write_pairs_jsonl(tmp_path / "crk-en.jsonl", [{"crk": "namoya", "en": "no way friend"}])
    _write_pairs_jsonl(tmp_path / "nah-es.jsonl", [{"nah": "amo", "es": "no tengo nada"}])
    monkeypatch.setattr(corpora, "INDIGENOUS_PANEL_LOCAL_DIR", str(tmp_path))

    class FakeCounter:
        def __init__(self, model, rate_limiter, api_key=None):
            pass

        def count(self, text):
            return _fake_word_count_fn(text)

    monkeypatch.setattr(claude_evaluate, "ClaudeTokenCounter", FakeCounter)

    out_path = tmp_path / "results.json"
    all_results = claude_evaluate.main([
        "--model", "claude-fake-model",
        "--eval-data-source", "indigenous_panel",
        "--output", str(out_path),
    ])

    result = all_results["claude-fake-model"]
    assert set(result) >= {"combined", "token_parity_by_anchor", "morphology_spread"}
    assert set(result["token_parity_by_anchor"]) == {"en", "es"}

    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded == all_results
