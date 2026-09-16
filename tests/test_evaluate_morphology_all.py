"""Tests for scripts.evaluate_morphology_all's own orchestration logic
(looping over a heterogeneous-CLI multi-system YAML config, per-system error
isolation, the claude_tokenizer/fairtok/flexitokens hard-exclusion guard,
merging into one combined file) -- evaluate_cli.main itself is monkeypatched
throughout so these stay fast/offline, not exercising any real tokenizer
checkpoint loading (that's covered by each system's own evaluate.py tests)."""

import json

import pytest

import evaluate as evaluate_cli
from scripts.evaluate_morphology_all import main, run_morphology_all


def _write_fake_result(output_path, result_key, value):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({result_key: value}, f)


def test_run_morphology_all_calls_each_system_with_its_own_args_plus_shared_flags(tmp_path, monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        name, output_path = argv[0], argv[-1]
        _write_fake_result(output_path, name, {"morphology": {"deu_Latn": {"med": 0.5}}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    cfg = {
        "output_dir": str(tmp_path),
        "morphology_gold_dir": "data/morphology_gold",
        "systems": [
            {"name": "bpe", "args": ["--checkpoint", "checkpoints/bpe_50k.json", "--eval-data-source", "bouquet"]},
            {"name": "hf_frontier", "args": ["--hf-repo-id", "gpt2", "--eval-data-source", "bouquet"]},
        ],
    }
    per_system_paths, failed = run_morphology_all(cfg)

    assert failed == {}
    assert per_system_paths == [f"{tmp_path}/bpe_morphology.json", f"{tmp_path}/hf_frontier_morphology.json"]
    # No --result-key anywhere -- hf_frontier doesn't support it, so it's
    # never passed for any system (see the module docstring).
    assert calls[0] == [
        "bpe", "--checkpoint", "checkpoints/bpe_50k.json", "--eval-data-source", "bouquet",
        "--morphology-gold-dir", "data/morphology_gold",
        "--output", f"{tmp_path}/bpe_morphology.json",
    ]
    assert calls[1] == [
        "hf_frontier", "--hf-repo-id", "gpt2", "--eval-data-source", "bouquet",
        "--morphology-gold-dir", "data/morphology_gold",
        "--output", f"{tmp_path}/hf_frontier_morphology.json",
    ]


@pytest.mark.parametrize("excluded_name", ["claude_tokenizer", "fairtok", "flexitokens"])
def test_run_morphology_all_rejects_excluded_systems(tmp_path, excluded_name):
    cfg = {
        "output_dir": str(tmp_path),
        "morphology_gold_dir": "data/morphology_gold",
        "systems": [{"name": excluded_name, "args": []}],
    }
    with pytest.raises(ValueError, match=excluded_name):
        run_morphology_all(cfg)


def test_run_morphology_all_isolates_failures(tmp_path, monkeypatch):
    def fake_main(argv):
        name, output_path = argv[0], argv[-1]
        if name == "blt":
            raise RuntimeError("gated repo access denied")
        _write_fake_result(output_path, name, {"morphology": {}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    cfg = {
        "output_dir": str(tmp_path),
        "morphology_gold_dir": "data/morphology_gold",
        "systems": [
            {"name": "bpe", "args": ["--checkpoint", "checkpoints/bpe_50k.json"]},
            {"name": "blt", "args": ["--device", "cpu"]},
            {"name": "manta", "args": ["--checkpoint", "checkpoints/manta_123.pt"]},
        ],
    }
    per_system_paths, failed = run_morphology_all(cfg)

    assert per_system_paths == [f"{tmp_path}/bpe_morphology.json", f"{tmp_path}/manta_morphology.json"]
    assert set(failed) == {"blt"}
    assert "gated repo" in failed["blt"]


def test_run_morphology_all_skips_system_with_existing_output(tmp_path, monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        name, output_path = argv[0], argv[-1]
        _write_fake_result(output_path, name, {"morphology": {}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    bpe_output = tmp_path / "bpe_morphology.json"
    _write_fake_result(bpe_output, "bpe", {"morphology": {"pre_existing": True}})

    cfg = {
        "output_dir": str(tmp_path),
        "morphology_gold_dir": "data/morphology_gold",
        "systems": [
            {"name": "bpe", "args": ["--checkpoint", "checkpoints/bpe_50k.json"]},
            {"name": "fanta", "args": ["--checkpoint", "checkpoints/fanta_6284655.pt"]},
        ],
    }
    per_system_paths, failed = run_morphology_all(cfg)

    assert calls == ["fanta"]
    assert failed == {}
    assert per_system_paths == [str(bpe_output), f"{tmp_path}/fanta_morphology.json"]
    with open(bpe_output) as f:
        assert json.load(f)["bpe"] == {"morphology": {"pre_existing": True}}


def test_main_combines_successful_systems_and_records_failures_under_failed_key(tmp_path, monkeypatch):
    def fake_main(argv):
        name, output_path = argv[0], argv[-1]
        if name == "magnet":
            raise ValueError("bad checkpoint")
        _write_fake_result(output_path, name, {"morphology": {"deu_Latn": {"med": 1.0}}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"output_dir: {tmp_path}\n"
        f"combined_output: {tmp_path}/combined.json\n"
        "morphology_gold_dir: data/morphology_gold\n"
        "systems:\n"
        "  - name: bpe\n"
        "    args: [\"--checkpoint\", \"checkpoints/bpe_50k.json\"]\n"
        "  - name: magnet\n"
        "    args: [\"--checkpoint\", \"checkpoints/magnet_bad.pt\"]\n"
    )

    main(["-c", str(config_path)])

    with open(tmp_path / "combined.json") as f:
        combined = json.load(f)
    assert combined["bpe"] == {"morphology": {"deu_Latn": {"med": 1.0}}}
    assert "magnet" not in combined
    assert "bad checkpoint" in combined["_failed"]["magnet"]


def test_main_reports_and_returns_early_if_every_system_fails(tmp_path, monkeypatch, capsys):
    def always_fails(argv):
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(evaluate_cli, "main", always_fails)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"output_dir: {tmp_path}\n"
        "morphology_gold_dir: data/morphology_gold\n"
        "systems:\n  - name: bpe\n    args: [\"--checkpoint\", \"nope.json\"]\n"
    )

    main(["-c", str(config_path)])

    assert not (tmp_path / "morphology_all.json").exists()
    assert "no systems succeeded" in capsys.readouterr().out
