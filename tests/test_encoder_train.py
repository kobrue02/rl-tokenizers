"""End-to-end regression test for systems.pretraining.encoder_train, mirroring
test_train.py's own tiny_shard_dir/bpe_checkpoint fixture pattern (same real
data_prep pipeline, same offline synthetic-source convention) -- the decoder
and this encoder train from the identical shard format, so this reuses that
exact fixture shape rather than a new one."""

import dataclasses
import os
import time

import torch

import systems.pretraining.encoder_train as encoder_train_module
from systems.pretraining.data_prep import prep_dataset
from systems.pretraining.encoder_train import EncoderTrainConfig, train
from systems.tokenization.bpe.model import fit_bpe

import pytest


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


def test_end_to_end_encoder_training_run_with_validation_and_grad_accum(tiny_shard_dir, tmp_path):
    """Exercises the whole loop together: train/val shard splitting
    (reused from train.py), periodic validate() calls, grad_accum_steps > 1,
    and a checkpoint save/resume round-trip."""
    cfg = EncoderTrainConfig(
        encoder_size="tiny",
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

    cfg_resumed = dataclasses.replace(
        cfg, total_steps=cfg.total_steps + 4, resume_from=os.path.join(cfg.output_dir, "final.pt")
    )
    train(cfg_resumed)

    assert os.path.exists(os.path.join(cfg.output_dir, "final.pt"))
    ckpt_resumed = torch.load(os.path.join(cfg.output_dir, "final.pt"), weights_only=False)
    assert ckpt_resumed["step"] == cfg_resumed.total_steps


def test_tok_per_sec_excludes_validate_and_checkpoint_overhead(tiny_shard_dir, tmp_path, monkeypatch, capsys):
    """See test_train.py's identical test for the full rationale -- this is
    the same bug, in encoder_train.py's copy of the identical logging
    pattern: t_last_log used to reset right after printing, BEFORE
    validate()/save_checkpoint() ran, so their wall time silently bled into
    the FOLLOWING window's dt. Injects an artificial delay into both and
    confirms the next step's reported tok/s isn't deflated by it."""
    real_validate = encoder_train_module.validate
    real_save_checkpoint = encoder_train_module.save_checkpoint

    def slow_validate(*args, **kwargs):
        time.sleep(0.3)
        return real_validate(*args, **kwargs)

    def slow_save_checkpoint(*args, **kwargs):
        time.sleep(0.3)
        return real_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(encoder_train_module, "validate", slow_validate)
    monkeypatch.setattr(encoder_train_module, "save_checkpoint", slow_save_checkpoint)

    cfg = EncoderTrainConfig(
        encoder_size="tiny",
        shard_dir=tiny_shard_dir,
        seq_len=16,
        per_device_batch_size=4,
        total_steps=4,
        val_fraction=0.5,
        eval_interval=1,
        log_steps=1,
        save_steps=2,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        dtype="float32",
        num_workers=0,
    )

    train(cfg)

    out = capsys.readouterr().out
    tok_per_sec_values = [
        float(line.split("tok/s=")[1].replace(",", ""))
        for line in out.splitlines()
        if "tok/s=" in line
    ]
    assert len(tok_per_sec_values) == cfg.total_steps
    baseline = tok_per_sec_values[0]
    for value in tok_per_sec_values[1:]:
        assert value > baseline * 0.1, tok_per_sec_values


def test_training_actually_reduces_loss(tiny_shard_dir, tmp_path, capsys):
    """Not just "doesn't crash" -- a real, if short, run should meaningfully
    reduce MLM loss from its (near-random, ln(vocab_size)-ish) starting
    point. Reads the printed per-step loss lines rather than instrumenting
    train() with a return value, matching this project's existing
    print-based training-loop convention (see train.py's own logging)."""
    cfg = EncoderTrainConfig(
        encoder_size="tiny",
        shard_dir=tiny_shard_dir,
        seq_len=16,
        per_device_batch_size=8,
        total_steps=60,
        learning_rate=5e-3,
        val_fraction=0,
        eval_interval=0,
        log_steps=1,
        save_steps=0,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        dtype="float32",
        num_workers=0,
        seed=0,
    )

    train(cfg)

    out = capsys.readouterr().out
    losses = [float(line.split("loss=")[1].split()[0]) for line in out.splitlines() if "loss=" in line]
    assert len(losses) == cfg.total_steps
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg
