"""Tests for systems.pretraining.train: checkpoint-rotation-across-resume,
train/val shard splitting, the validate() loss pass, and an end-to-end
training run exercising all of it together.

_existing_checkpoints seeds the rotation bookkeeping from whatever's
already on disk at startup, so keep_last_n_checkpoints correctly rotates
across a --resume-from restart, not just within one process's own
lifetime -- see that function's own docstring for the real disk-quota
incident this guards against."""

import dataclasses
import os

import pytest
import torch

from systems.pretraining.data_prep import prep_dataset
from systems.pretraining.model import TransformerLM
from systems.pretraining.model_configs import get_preset
from systems.pretraining.shard_dataset import ShardedTokenDataset
from systems.pretraining.train import (
    TrainConfig,
    _existing_checkpoints,
    _split_train_val_shard_files,
    train,
    validate,
)
from systems.tokenization.bpe.model import fit_bpe


def test_returns_empty_list_when_dir_missing(tmp_path):
    assert _existing_checkpoints(str(tmp_path / "does_not_exist")) == []


def test_finds_and_sorts_step_checkpoints_by_step_number(tmp_path):
    for step in (3000, 1000, 2000):
        (tmp_path / f"step_{step}.pt").write_text("")

    result = _existing_checkpoints(str(tmp_path))

    assert result == [
        str(tmp_path / "step_1000.pt"),
        str(tmp_path / "step_2000.pt"),
        str(tmp_path / "step_3000.pt"),
    ]


def test_excludes_final_pt(tmp_path):
    (tmp_path / "step_1000.pt").write_text("")
    (tmp_path / "final.pt").write_text("")

    assert _existing_checkpoints(str(tmp_path)) == [str(tmp_path / "step_1000.pt")]


def test_ignores_files_that_dont_parse_as_step_int(tmp_path):
    (tmp_path / "step_1000.pt").write_text("")
    (tmp_path / "step_abc.pt").write_text("")
    (tmp_path / "step_.pt").write_text("")
    (tmp_path / "not_a_checkpoint.txt").write_text("")

    assert _existing_checkpoints(str(tmp_path)) == [str(tmp_path / "step_1000.pt")]


def test_empty_dir_returns_empty_list(tmp_path):
    assert _existing_checkpoints(str(tmp_path)) == []


def test_split_train_val_disabled_when_val_fraction_non_positive():
    shard_files = [f"shard_{i:05d}.bin" for i in range(10)]
    train_files, val_files = _split_train_val_shard_files(shard_files, 0.0)
    assert train_files == shard_files
    assert val_files == []


def test_split_train_val_disabled_when_too_few_shards():
    train_files, val_files = _split_train_val_shard_files(["shard_00000.bin"], 0.2)
    assert train_files == ["shard_00000.bin"]
    assert val_files == []


def test_split_train_val_uses_a_stride_not_just_the_tail():
    """The real bug this guards against: glot500's per-language corpus
    sizes span orders of magnitude, and languages drop out of the
    round-robin once their own data is exhausted -- reserving only the
    LAST few shards for validation would skew toward whichever languages
    had the most data, rather than sampling representatively across the
    whole corpus."""
    shard_files = [f"shard_{i:05d}.bin" for i in range(10)]
    train_files, val_files = _split_train_val_shard_files(shard_files, 0.2)

    assert val_files == ["shard_00000.bin", "shard_00005.bin"]
    assert train_files == [
        "shard_00001.bin", "shard_00002.bin", "shard_00003.bin", "shard_00004.bin",
        "shard_00006.bin", "shard_00007.bin", "shard_00008.bin", "shard_00009.bin",
    ]
    assert set(train_files).isdisjoint(val_files)
    assert set(train_files) | set(val_files) == set(shard_files)


def test_split_train_val_stays_disjoint_and_complete_across_fractions():
    shard_files = [f"shard_{i:05d}.bin" for i in range(20)]
    for val_fraction in (0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
        train_files, val_files = _split_train_val_shard_files(shard_files, val_fraction)
        assert set(train_files).isdisjoint(val_files)
        assert set(train_files) | set(val_files) == set(shard_files)
        assert train_files  # never empty -- see this function's own docstring


@pytest.fixture
def bpe_checkpoint(tmp_path):
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a small tokenizer trained only for this test",
        "held out validation should never leak into training samples",
    ]
    model = fit_bpe(sentences, vocab_size=300)
    path = tmp_path / "bpe_checkpoint.json"
    model.tokenizer.save(str(path))
    return str(path)


@pytest.fixture
def tiny_shard_dir(tmp_path, bpe_checkpoint):
    """A real, tiny packed-shard corpus (via the actual data_prep pipeline,
    same offline synthetic-source convention as test_data_prep.py) small
    enough to produce several shards so train/val splitting has something
    real to split."""
    out_dir = tmp_path / "shards"
    prep_dataset(
        dataset_name="synthetic",
        system="bpe",
        checkpoint_path=bpe_checkpoint,
        output_dir=str(out_dir),
        max_docs=400,
        dedup=False,
        shard_size=200,  # small on purpose -- forces multiple shards
        encode_batch_size=8,
        bucket_pool_multiplier=1,
    )
    return str(out_dir)


def test_validate_returns_finite_average_loss_over_a_fixed_val_set(tiny_shard_dir):
    model_cfg = get_preset("tiny")
    dataset = ShardedTokenDataset(tiny_shard_dir, seq_len=16, num_samples=8, seed=0)
    from systems.pretraining.shard_dataset import load_shard_meta
    vocab_size = load_shard_meta(tiny_shard_dir)["vocab_size"]
    model = TransformerLM(model_cfg, vocab_size)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    loss = validate(model, val_loader, torch.device("cpu"), torch.float32)

    assert torch.isfinite(torch.tensor(loss))


def test_validate_restores_training_mode_if_model_was_training(tiny_shard_dir):
    model_cfg = get_preset("tiny")
    from systems.pretraining.shard_dataset import load_shard_meta
    vocab_size = load_shard_meta(tiny_shard_dir)["vocab_size"]
    model = TransformerLM(model_cfg, vocab_size)
    model.train()
    dataset = ShardedTokenDataset(tiny_shard_dir, seq_len=16, num_samples=8, seed=0)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    validate(model, val_loader, torch.device("cpu"), torch.float32)

    assert model.training


def test_end_to_end_training_run_with_validation_and_grad_accum(tiny_shard_dir, tmp_path):
    """Exercises everything touched this round together: train/val shard
    splitting, periodic validate() calls, grad_accum_steps > 1 (the
    no_sync() code path is only meaningful under real DDP, but this at
    least confirms the non-distributed contextlib.nullcontext() fallback
    doesn't break anything), and a checkpoint save/resume round-trip with
    the new TrainConfig fields present."""
    cfg = TrainConfig(
        model_size="tiny",
        shard_dir=tiny_shard_dir,
        seq_len=16,
        per_device_batch_size=4,
        grad_accum_steps=2,
        total_steps=6,
        val_fraction=0.5,
        eval_interval=2,
        eval_iters=2,
        log_steps=1,
        save_steps=3,
        keep_last_n_checkpoints=1,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        dtype="float32",
        num_workers=0,
    )

    train(cfg)

    assert os.path.exists(os.path.join(cfg.output_dir, "final.pt"))
    ckpt = torch.load(os.path.join(cfg.output_dir, "final.pt"), weights_only=False)
    assert ckpt["step"] == cfg.total_steps

    # Resume from the final checkpoint for a few more steps -- confirms
    # save/load still round-trips correctly with the new fields present.
    cfg_resumed = dataclasses.replace(
        cfg, total_steps=cfg.total_steps + 4, resume_from=os.path.join(cfg.output_dir, "final.pt")
    )
    train(cfg_resumed)
    ckpt_resumed = torch.load(os.path.join(cfg.output_dir, "final.pt"), weights_only=False)
    assert ckpt_resumed["step"] == cfg_resumed.total_steps
