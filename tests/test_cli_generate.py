"""Tests for cli_generate.py's interactive "app" mode -- checkpoint
discovery, tokenizer auto-resolution from a checkpoint's own stored
shard_dir, mode-branching (interactive vs. batch/scripted), and the full
interactive loop end-to-end against a real (tiny) trained checkpoint. The
existing batch/scripted mode's own coverage is run_smoke_test() (unchanged,
still exercised directly in this file too, to confirm interactive mode
didn't regress it).
"""

import os

import pytest

from systems.pretraining import cli_generate
from systems.pretraining.data_prep import prep_dataset
from systems.pretraining.train import TrainConfig, train
from systems.tokenization.bpe.model import fit_bpe


def test_existing_batch_mode_smoke_test_still_passes():
    cli_generate.run_smoke_test()


def test_discover_checkpoints_is_stat_only_and_newest_first(tmp_path):
    import time

    run1 = tmp_path / "run1"
    run1.mkdir()
    run2 = tmp_path / "run2"
    run2.mkdir()
    older = run1 / "step_100.pt"
    older.write_text("not a real checkpoint")  # discover_checkpoints must never torch.load
    time.sleep(0.05)
    newer = run2 / "final.pt"
    newer.write_text("not a real checkpoint")

    found = cli_generate.discover_checkpoints(str(tmp_path))

    assert found == [str(newer), str(older)]


def test_discover_checkpoints_empty_dir_returns_empty_list(tmp_path):
    assert cli_generate.discover_checkpoints(str(tmp_path)) == []


def test_main_routes_to_interactive_when_no_prompt_given(monkeypatch):
    called = {}
    monkeypatch.setattr(cli_generate, "interactive_main", lambda args: called.setdefault("ran", True))

    cli_generate.main([])

    assert called.get("ran") is True


def test_main_batch_mode_requires_checkpoint_system_and_tokenizer_checkpoint():
    with pytest.raises(SystemExit, match=r"--checkpoint.*--system.*--tokenizer-checkpoint"):
        cli_generate.main(["--prompt", "hello"])


@pytest.fixture
def tiny_trained_checkpoint(tmp_path):
    """A real, tiny end-to-end trained checkpoint (same construction as
    tests/test_train.py's own fixtures) -- what interactive_main's
    auto-resolution actually needs to succeed against."""
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a small tokenizer trained only for this test",
        "held out validation should never leak into training",
    ]
    bpe_model = fit_bpe(sentences, vocab_size=300)
    bpe_ckpt = tmp_path / "bpe.json"
    bpe_model.tokenizer.save(str(bpe_ckpt))

    shard_dir = tmp_path / "shards"
    prep_dataset(
        dataset_name="synthetic", system="bpe", checkpoint_path=str(bpe_ckpt),
        output_dir=str(shard_dir), max_docs=400, dedup=False, shard_size=200,
        encode_batch_size=8, bucket_pool_multiplier=1,
    )

    out_dir = tmp_path / "out"
    cfg = TrainConfig(
        model_size="tiny", shard_dir=str(shard_dir), seq_len=16, per_device_batch_size=4,
        total_steps=2, val_fraction=0.0, eval_interval=0, log_steps=1, save_steps=0,
        output_dir=str(out_dir), device="cpu", dtype="float32", num_workers=0,
    )
    train(cfg)
    return str(out_dir / "final.pt")


def test_load_checkpoint_and_resolve_tokenizer_recovers_system_from_shard_dir(tiny_trained_checkpoint):
    model, system, tokenizer_checkpoint, step = cli_generate.load_checkpoint_and_resolve_tokenizer(
        tiny_trained_checkpoint, device="cpu"
    )

    assert system == "bpe"
    assert os.path.exists(tokenizer_checkpoint)
    assert step == 2
    assert model.training is False  # .eval() already applied


def test_interactive_main_end_to_end_discover_select_generate_quit(tiny_trained_checkpoint, monkeypatch, capsys):
    checkpoints_dir = os.path.dirname(os.path.dirname(tiny_trained_checkpoint))  # tmp_path itself

    responses = iter(["0", "the quick", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    args = cli_generate.build_arg_parser().parse_args(
        ["--checkpoints-dir", checkpoints_dir, "--max-new-tokens", "5"]
    )
    cli_generate.interactive_main(args)

    out = capsys.readouterr().out
    assert "Found 1 checkpoint(s)" in out
    assert "system='bpe'" in out
    assert "Ready -- generating" in out


def test_interactive_main_skips_browse_step_when_checkpoint_given_directly(tiny_trained_checkpoint, monkeypatch, capsys):
    responses = iter(["quit"])  # no checkpoint-selection prompt expected
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    args = cli_generate.build_arg_parser().parse_args(
        ["--checkpoint", tiny_trained_checkpoint, "--max-new-tokens", "5"]
    )
    cli_generate.interactive_main(args)

    out = capsys.readouterr().out
    assert "Found" not in out  # browse step never ran
    assert "system='bpe'" in out
