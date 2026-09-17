"""Tests for scripts.check_pipeline_status's own classification logic --
each stage checker is exercised directly against a tmp_path filesystem
fixture, not the real repo, so these stay fast/deterministic regardless of
what's actually been run on the cluster."""

import json
import os

import pytest

from scripts.check_pipeline_status import (
    _check_output_file,
    _check_prep,
    _check_pretrain,
    _check_wrapper,
    _classify_module,
    _module_target,
    check_tokenizer_training,
)


def test_module_target_parses_first_line_comment(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("# systems/pretraining/data_prep.py -- sbatch jobs/prep/foo.sh\nkey: value\n")
    assert _module_target(str(config_path)) == "systems/pretraining/data_prep.py"


def test_module_target_returns_none_without_the_convention(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("key: value\n")
    assert _module_target(str(config_path)) is None


@pytest.mark.parametrize("module,expected_stage", [
    ("systems/pretraining/data_prep.py", "prep"),
    ("systems/pretraining/cli.py", "pretrain"),
    ("systems/pretraining/encoder_cli.py", "pretrain"),
    ("systems/pretraining/cli_generate.py", "generate"),
    ("systems/pretraining/cli_eval.py", "eval"),
    ("systems/pretraining/cli_lm_eval.py", "eval"),
    ("systems/tokenization/bpe/cli.py", "train_tokenizer"),
    ("systems/tokenization/hf_frontier/evaluate.py", "eval"),
    ("scripts/evaluate_own_tokenizers_indigenous_panel.py", "eval"),
    ("scripts/evaluate_morphology_all.py", "eval"),
    ("scripts/some_unregistered_wrapper.py", "unknown"),
])
def test_classify_module(module, expected_stage):
    stage, _checker = _classify_module(module)
    assert stage == expected_stage


def test_check_prep_missing_when_output_dir_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    status, detail = _check_prep({"output_dir": "pretrain_data/nope"})
    assert status == "missing"
    assert "does not exist" in detail


def test_check_prep_in_progress_with_resumable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("pretrain_data/foo")
    with open("pretrain_data/foo/prep_checkpoint.json", "w") as f:
        json.dump({}, f)
    status, detail = _check_prep({"output_dir": "pretrain_data/foo"})
    assert status == "in_progress"
    assert "resumable" in detail


def test_check_prep_done_reports_shard_count_and_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("pretrain_data/foo")
    with open("pretrain_data/foo/shards_meta.json", "w") as f:
        json.dump({"shard_files": ["a.bin", "b.bin"], "total_tokens": 5_000_000_000}, f)
    status, detail = _check_prep({"output_dir": "pretrain_data/foo"})
    assert status == "done"
    assert "2 shards" in detail
    assert "5,000,000,000 tokens" in detail


def test_check_pretrain_done_when_final_pt_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("checkpoints/pretrain_foo")
    open("checkpoints/pretrain_foo/final.pt", "w").close()
    status, detail = _check_pretrain({"output_dir": "checkpoints/pretrain_foo"})
    assert status == "done"
    assert "final.pt" in detail


def test_check_pretrain_in_progress_reports_latest_step_of_total(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("checkpoints/pretrain_foo")
    open("checkpoints/pretrain_foo/step_1000.pt", "w").close()
    open("checkpoints/pretrain_foo/step_5000.pt", "w").close()
    status, detail = _check_pretrain({"output_dir": "checkpoints/pretrain_foo", "total_steps": 250000})
    assert status == "in_progress"
    assert "step_5000.pt/250000" in detail  # picks the LATEST step, not the first found


def test_check_pretrain_defaults_output_dir_to_checkpoints_pretrain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    status, detail = _check_pretrain({})
    assert status == "missing"
    assert "checkpoints/pretrain " in detail or "checkpoints/pretrain has" in detail


def test_check_output_file_done_and_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _check_output_file({"output": "results/nope.json"})[0] == "missing"
    os.makedirs("results")
    open("results/here.json", "w").close()
    assert _check_output_file({"output": "results/here.json"})[0] == "done"


def test_check_output_file_unknown_when_key_absent():
    status, detail = _check_output_file({})
    assert status == "unknown"


def test_check_wrapper_exact_suffix_not_loose_glob(tmp_path, monkeypatch):
    """A pre-existing FILE sharing the {name}_ prefix but the WRONG suffix
    (e.g. bpe_comparison.json from an unrelated normal eval run) must not
    count as this wrapper's own bpe_morphology.json being done -- this is
    the exact false-positive this script's own history caught (see the
    module's _WRAPPER_SUFFIXES comment)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("results")
    open("results/bpe_comparison.json", "w").close()  # wrong suffix, must NOT count
    cfg = {"output_dir": "results", "combined_output": "results/combined.json", "systems": [{"name": "bpe"}]}
    status, detail, per_system = _check_wrapper(cfg, suffix="morphology")
    assert per_system["bpe"] is None
    assert status == "missing"


def test_check_wrapper_reports_done_with_correct_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results")
    open("results/bpe_morphology.json", "w").close()
    open("results/combined.json", "w").close()
    cfg = {"output_dir": "results", "combined_output": "results/combined.json", "systems": [{"name": "bpe"}]}
    status, detail, per_system = _check_wrapper(cfg, suffix="morphology")
    assert per_system["bpe"] == "bpe_morphology.json"
    assert status == "done"


def test_check_wrapper_in_progress_when_some_systems_done_but_no_combined(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results")
    open("results/bpe_morphology.json", "w").close()
    cfg = {
        "output_dir": "results", "combined_output": "results/combined.json",
        "systems": [{"name": "bpe"}, {"name": "fanta"}],
    }
    status, detail, per_system = _check_wrapper(cfg, suffix="morphology")
    assert status == "in_progress"
    assert per_system == {"bpe": "bpe_morphology.json", "fanta": None}


def test_check_tokenizer_training_is_independent_of_any_config_file(tmp_path, monkeypatch):
    """magnet/manta have no configs/train_tokenizer/*.yml at all (trained
    directly via their own .sh scripts with job-ID-tagged paths) -- this
    must still report their status via the checkpoint glob."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("checkpoints")
    open("checkpoints/magnet_123456.pt", "w").close()
    rows = check_tokenizer_training()
    by_name = {r["name"]: r for r in rows}
    assert by_name["magnet"]["status"] == "done"
    assert by_name["manta"]["status"] == "missing"
    assert set(by_name) == {"fairtok", "magnet", "flexitokens", "manta", "fanta", "superbpe", "bpe", "parity_bpe"}
