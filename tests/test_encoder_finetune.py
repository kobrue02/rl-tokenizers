"""Tests for encoder_finetune.load_finetune_model: the pretrained-encoder-
body transplant into a fresh task head (token or sequence classification)."""

import dataclasses

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

from systems.pretraining.encoder_finetune import load_finetune_model
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_train import EncoderTrainConfig


def _save_mlm_checkpoint(tmp_path, vocab_size=200):
    cfg = EncoderTrainConfig(encoder_size="tiny", shard_dir="unused", seq_len=16)
    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)
    mlm_model = build_encoder(preset, vocab_size)
    path = tmp_path / "mlm_checkpoint.pt"
    torch.save(
        {
            "step": 10,
            "model_state_dict": mlm_model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "config": dataclasses.asdict(cfg),
            "vocab_size": vocab_size,
        },
        path,
    )
    return str(path), mlm_model


def test_token_classification_transplants_pretrained_encoder_body(tmp_path):
    checkpoint_path, mlm_model = _save_mlm_checkpoint(tmp_path)

    task_model = load_finetune_model(
        checkpoint_path, AutoModelForTokenClassification, device="cpu", num_labels=7
    )

    assert task_model.config.num_labels == 7
    for (name, p_mlm), (_, p_task) in zip(
        mlm_model.roberta.named_parameters(), task_model.roberta.named_parameters()
    ):
        assert torch.equal(p_mlm, p_task), name
    # Head is freshly initialized, not part of the MLM checkpoint at all --
    # just confirm its shape matches num_labels (there's no "correct"
    # pretrained value to compare against).
    assert task_model.classifier.out_features == 7


def test_sequence_classification_transplants_pretrained_encoder_body(tmp_path):
    checkpoint_path, mlm_model = _save_mlm_checkpoint(tmp_path)

    task_model = load_finetune_model(
        checkpoint_path, AutoModelForSequenceClassification, device="cpu", num_labels=6
    )

    assert task_model.config.num_labels == 6
    for (name, p_mlm), (_, p_task) in zip(
        mlm_model.roberta.named_parameters(), task_model.roberta.named_parameters()
    ):
        assert torch.equal(p_mlm, p_task), name


def test_config_overrides_applied_beyond_num_labels(tmp_path):
    checkpoint_path, _ = _save_mlm_checkpoint(tmp_path)

    id2label = {0: "O", 1: "B-PER"}
    task_model = load_finetune_model(
        checkpoint_path, AutoModelForTokenClassification, device="cpu", num_labels=2, id2label=id2label,
    )

    assert task_model.config.id2label == id2label


def test_two_finetunes_from_the_same_checkpoint_get_independently_initialized_heads(tmp_path):
    """Confirms the task head really IS freshly randomly initialized each
    call, not accidentally shared/cached state leaking between finetuning
    runs off the same pretrained checkpoint."""
    checkpoint_path, _ = _save_mlm_checkpoint(tmp_path)

    model_a = load_finetune_model(checkpoint_path, AutoModelForTokenClassification, num_labels=7)
    model_b = load_finetune_model(checkpoint_path, AutoModelForTokenClassification, num_labels=7)

    assert not torch.equal(model_a.classifier.weight, model_b.classifier.weight)
