"""Tests for scripts.evaluate_own_tokenizers_indigenous_panel's own orchestration
logic (looping over a multi-system YAML config, per-system error isolation,
merging into one combined file) -- evaluate_cli.main itself is monkeypatched
throughout so these stay fast/offline, not exercising any real tokenizer
checkpoint loading (that's covered by each system's own evaluate.py tests)."""

import json

import pytest

import evaluate as evaluate_cli
import scripts.evaluate_own_tokenizers_indigenous_panel as own_tokenizers_script
from scripts.evaluate_own_tokenizers_indigenous_panel import main, run_own_tokenizers_indigenous_panel


def _write_fake_result(output_path, result_key, value):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({result_key: value}, f)


def test_run_own_tokenizers_indigenous_panel_calls_each_system_with_correct_args(tmp_path, monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        # argv shape: [name, "--checkpoint", path, "--eval-data-source", "indigenous_panel",
        #              "--output", output_path, "--result-key", name, *extra_args]
        name, output_path = argv[0], argv[6]
        _write_fake_result(output_path, name, {"combined": {"avg_compression": 1.0}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    cfg = {
        "output_dir": str(tmp_path),
        "systems": [
            {"name": "bpe", "checkpoint": "checkpoints/bpe_50k.json"},
            {"name": "fanta", "checkpoint": "checkpoints/fanta_6284655.pt", "extra_args": ["--num-groups", "5"]},
        ],
    }
    per_system_paths, failed = run_own_tokenizers_indigenous_panel(cfg)

    assert failed == {}
    assert per_system_paths == [
        f"{tmp_path}/bpe_indigenous_panel.json",
        f"{tmp_path}/fanta_indigenous_panel.json",
    ]
    assert calls[0] == [
        "bpe", "--checkpoint", "checkpoints/bpe_50k.json",
        "--eval-data-source", "indigenous_panel",
        "--output", f"{tmp_path}/bpe_indigenous_panel.json",
        "--result-key", "bpe",
    ]
    assert calls[1] == [
        "fanta", "--checkpoint", "checkpoints/fanta_6284655.pt",
        "--eval-data-source", "indigenous_panel",
        "--output", f"{tmp_path}/fanta_indigenous_panel.json",
        "--result-key", "fanta",
        "--num-groups", "5",
    ]


def test_run_own_tokenizers_indigenous_panel_isolates_failures(tmp_path, monkeypatch):
    """One system raising (e.g. a stale/placeholder --checkpoint) must not
    stop the rest of the list from running -- same per-entry isolation as
    hf_frontier/evaluate.py's own per-repo loop."""

    def fake_main(argv):
        name, output_path = argv[0], argv[6]
        if name == "magnet":
            raise FileNotFoundError("checkpoints/magnet_<FILL_IN_JOB_ID>.pt")
        _write_fake_result(output_path, name, {"combined": {"avg_compression": 1.0}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    cfg = {
        "output_dir": str(tmp_path),
        "systems": [
            {"name": "bpe", "checkpoint": "checkpoints/bpe_50k.json"},
            {"name": "magnet", "checkpoint": "checkpoints/magnet_<FILL_IN_JOB_ID>.pt"},
            {"name": "manta", "checkpoint": "checkpoints/manta_123.pt"},
        ],
    }
    per_system_paths, failed = run_own_tokenizers_indigenous_panel(cfg)

    assert per_system_paths == [f"{tmp_path}/bpe_indigenous_panel.json", f"{tmp_path}/manta_indigenous_panel.json"]
    assert set(failed) == {"magnet"}
    assert "FILL_IN_JOB_ID" in failed["magnet"]


def test_main_combines_successful_systems_and_records_failures_under_failed_key(tmp_path, monkeypatch):
    def fake_main(argv):
        name, output_path = argv[0], argv[6]
        if name == "flexitokens":
            raise ValueError("bad checkpoint")
        _write_fake_result(output_path, name, {"combined": {"avg_compression": 2.0}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"output_dir: {tmp_path}\n"
        f"combined_output: {tmp_path}/combined.json\n"
        "systems:\n"
        "  - name: bpe\n"
        "    checkpoint: checkpoints/bpe_50k.json\n"
        "  - name: flexitokens\n"
        "    checkpoint: checkpoints/flexitokens_bad.pt\n"
    )

    main(["-c", str(config_path)])

    with open(tmp_path / "combined.json") as f:
        combined = json.load(f)
    assert combined["bpe"] == {"combined": {"avg_compression": 2.0}}
    assert "flexitokens" not in combined
    assert "bad checkpoint" in combined["_failed"]["flexitokens"]


def test_run_own_tokenizers_indigenous_panel_skips_system_with_existing_output(tmp_path, monkeypatch):
    """A resubmit after a partial failure (e.g. the real OOM this was written
    for) must not waste time re-evaluating a system that already succeeded --
    same "don't redo completed work" fix as superbpe's own stage1_result.json."""
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        name, output_path = argv[0], argv[6]
        _write_fake_result(output_path, name, {"combined": {"avg_compression": 1.0}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    bpe_output = tmp_path / "bpe_indigenous_panel.json"
    _write_fake_result(bpe_output, "bpe", {"combined": {"avg_compression": 9.0}})  # pre-existing "completed" result

    cfg = {
        "output_dir": str(tmp_path),
        "systems": [
            {"name": "bpe", "checkpoint": "checkpoints/bpe_50k.json"},
            {"name": "fanta", "checkpoint": "checkpoints/fanta_6284655.pt"},
        ],
    }
    per_system_paths, failed = run_own_tokenizers_indigenous_panel(cfg)

    assert calls == ["fanta"]  # bpe skipped entirely, never re-invoked
    assert failed == {}
    assert per_system_paths == [str(bpe_output), f"{tmp_path}/fanta_indigenous_panel.json"]
    with open(bpe_output) as f:
        assert json.load(f)["bpe"]["combined"]["avg_compression"] == 9.0  # untouched


def test_run_own_tokenizers_indigenous_panel_force_redoes_existing_output(tmp_path, monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        name, output_path = argv[0], argv[6]
        _write_fake_result(output_path, name, {"combined": {"avg_compression": 1.0}})

    monkeypatch.setattr(evaluate_cli, "main", fake_main)

    bpe_output = tmp_path / "bpe_indigenous_panel.json"
    _write_fake_result(bpe_output, "bpe", {"combined": {"avg_compression": 9.0}})

    cfg = {"output_dir": str(tmp_path), "systems": [{"name": "bpe", "checkpoint": "checkpoints/bpe_50k.json"}]}
    run_own_tokenizers_indigenous_panel(cfg, force=True)

    assert calls == ["bpe"]
    with open(bpe_output) as f:
        assert json.load(f)["bpe"]["combined"]["avg_compression"] == 1.0  # overwritten


def test_main_reports_and_returns_early_if_every_system_fails(tmp_path, monkeypatch, capsys):
    def always_fails(argv):
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(evaluate_cli, "main", always_fails)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"output_dir: {tmp_path}\nsystems:\n  - name: bpe\n    checkpoint: nope.json\n"
    )

    main(["-c", str(config_path)])

    assert not (tmp_path / "own_tokenizers_indigenous_panel.json").exists()
    assert "no systems succeeded" in capsys.readouterr().out
