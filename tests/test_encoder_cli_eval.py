"""Smoke tests for encoder_cli_eval.py's wiring: checkpoint round-trip via
load_pretrained_encoder, and each --benchmark dispatch path against a
monkeypatched common.data.corpora.stream_groups (no real network/data
dependency -- same convention test_data_prep.py's own monkeypatched
`broken_stream` tests use)."""

import dataclasses

import torch

from systems.pretraining.encoder_cli_eval import (
    _load_vocab,
    load_pretrained_encoder,
    main,
    run_pppl,
    run_retrieval,
    run_roundtrip,
)
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_train import EncoderTrainConfig
from systems.tokenization.bpe.model import fit_bpe


def _bpe_checkpoint(tmp_path):
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a small tokenizer trained only for this test",
    ]
    model = fit_bpe(sentences, vocab_size=300)
    path = tmp_path / "bpe_checkpoint.json"
    model.tokenizer.save(str(path))
    return str(path)


def _save_tiny_checkpoint(tmp_path, vocab_size=300):
    cfg = EncoderTrainConfig(encoder_size="tiny", shard_dir="unused", seq_len=16)
    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)
    model = build_encoder(preset, vocab_size)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "config": dataclasses.asdict(cfg),
            "vocab_size": vocab_size,
        },
        path,
    )
    return str(path), model


def test_load_pretrained_encoder_round_trips_weights(tmp_path):
    checkpoint_path, original_model = _save_tiny_checkpoint(tmp_path)

    loaded = load_pretrained_encoder(checkpoint_path, device="cpu")

    for (name, p_orig), (_, p_loaded) in zip(
        original_model.named_parameters(), loaded.named_parameters()
    ):
        assert torch.equal(p_orig, p_loaded), name


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_run_retrieval_dispatches_tatoeba_mt_and_reports_topk(tmp_path, monkeypatch, capsys):
    bpe_checkpoint = _bpe_checkpoint(tmp_path)
    vocab = _load_vocab("bpe", bpe_checkpoint, "")
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size=vocab.vocab_size)
    model = load_pretrained_encoder(checkpoint_path, device="cpu")

    import systems.pretraining.encoder_cli_eval as cli_eval_module

    fake_rows = [
        {"deu": "der schnelle Fuchs", "eng": "the quick fox"},
        {"deu": "ein kleiner Test", "eng": "a small test"},
    ]
    monkeypatch.setattr(cli_eval_module, "stream_groups", lambda *a, **k: iter(fake_rows))

    args = _Args(dataset="tatoeba_mt", split="test", pair="deu-eng", device="cpu", layer=1)
    run_retrieval(args, model, vocab)

    out = capsys.readouterr().out
    assert "retrieval: 2 deu-eng pairs" in out
    assert "top1_accuracy=" in out


def test_run_roundtrip_dispatches_bible_nlp(tmp_path, monkeypatch, capsys):
    bpe_checkpoint = _bpe_checkpoint(tmp_path)
    vocab = _load_vocab("bpe", bpe_checkpoint, "")
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size=vocab.vocab_size)
    model = load_pretrained_encoder(checkpoint_path, device="cpu")

    import systems.pretraining.encoder_cli_eval as cli_eval_module

    fake_rows = [{"eng": "a b c", "fra": "a b c", "deu": "a b c"}]
    monkeypatch.setattr(cli_eval_module, "stream_groups", lambda *a, **k: iter(fake_rows))

    args = _Args(cycle_langs="eng,fra,deu,eng", device="cpu", layer=1)
    run_roundtrip(args, model, vocab)

    out = capsys.readouterr().out
    assert "roundtrip: 1 verse groups" in out
    assert "roundtrip_accuracy=" in out


def test_run_pppl_dispatches_tatoeba_mt(tmp_path, monkeypatch, capsys):
    bpe_checkpoint = _bpe_checkpoint(tmp_path)
    vocab = _load_vocab("bpe", bpe_checkpoint, "")
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size=vocab.vocab_size)
    model = load_pretrained_encoder(checkpoint_path, device="cpu")

    import systems.pretraining.encoder_cli_eval as cli_eval_module

    fake_rows = [{"deu": "der schnelle Fuchs", "eng": "the quick fox"}]
    monkeypatch.setattr(cli_eval_module, "stream_groups", lambda *a, **k: iter(fake_rows))

    args = _Args(dataset="tatoeba_mt", split="test", pair="deu-eng", lang="deu", device="cpu")
    run_pppl(args, model, vocab)

    out = capsys.readouterr().out
    assert "pppl: 1 deu sentences" in out
    assert "pseudoperplexity=" in out


def test_main_with_use_wandb_runs_end_to_end_for_pppl(tmp_path, monkeypatch, capsys):
    """--use-wandb must flow all the way through main() -> run_pppl and log
    the result. WANDB_MODE=disabled makes wandb.init() a documented no-op
    (NoopRun, no network/login needed) -- confirmed live, same reasoning as
    test_encoder_finetune_tagging.py's identical test."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    bpe_checkpoint = _bpe_checkpoint(tmp_path)
    vocab = _load_vocab("bpe", bpe_checkpoint, "")
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size=vocab.vocab_size)

    import systems.pretraining.encoder_cli_eval as cli_eval_module

    fake_rows = [{"deu": "der schnelle Fuchs", "eng": "the quick fox"}]
    monkeypatch.setattr(cli_eval_module, "stream_groups", lambda *a, **k: iter(fake_rows))

    main([
        "--checkpoint", checkpoint_path,
        "--system", "bpe", "--tokenizer-checkpoint", bpe_checkpoint,
        "--benchmark", "pppl", "--dataset", "tatoeba_mt", "--split", "test", "--pair", "deu-eng", "--lang", "deu",
        "--device", "cpu",
        "--use-wandb", "--run-name", "test-cli-run",
    ])

    out = capsys.readouterr().out
    assert "logged eval results to wandb project=" in out
