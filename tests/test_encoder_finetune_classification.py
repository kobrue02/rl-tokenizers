"""End-to-end test for encoder_finetune_classification: real Trainer.train()
+ Trainer.evaluate() over a tiny synthetic in-memory Taxi1500-shaped
dataset (no real HF Hub dependency -- see the module's own docstring)."""

import dataclasses

import pytest
import torch

from systems.pretraining.encoder_finetune_classification import (
    ClassificationDataset,
    TAXI1500_LABELS,
    collate_classification_batch,
    finetune_classification,
)
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_tokenizer import EncoderVocab
from systems.pretraining.encoder_train import EncoderTrainConfig
from systems.pretraining.tokenizer_adapter import TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe

TRAIN_ROWS = [
    {"text": "you should follow this good advice", "label": 0},
    {"text": "we believe and have faith in god", "label": 1},
    {"text": "there was war and violence and death", "label": 2},
    {"text": "grace and mercy were given freely", "label": 3},
    {"text": "he sinned and did wrong against god", "label": 4},
    {"text": "the land was flat and full of trees", "label": 5},
]
EVAL_ROWS = [
    {"text": "follow this advice and be good", "label": 0},
    {"text": "faith in god gives us hope", "label": 1},
]


@pytest.fixture
def bpe_checkpoint(tmp_path):
    sentences = [row["text"] for row in TRAIN_ROWS + EVAL_ROWS] + [
        "a small tokenizer trained only for this test"
    ]
    model = fit_bpe(sentences, vocab_size=300)
    path = tmp_path / "bpe_checkpoint.json"
    model.tokenizer.save(str(path))
    return str(path)


@pytest.fixture
def vocab(bpe_checkpoint):
    return EncoderVocab(TokenizerAdapter.load("bpe", bpe_checkpoint))


@pytest.fixture
def mlm_checkpoint(tmp_path, vocab):
    cfg = EncoderTrainConfig(encoder_size="tiny", shard_dir="unused", seq_len=32)
    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)
    model = build_encoder(preset, vocab.vocab_size)
    path = tmp_path / "mlm_checkpoint.pt"
    torch.save(
        {
            "step": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "config": dataclasses.asdict(cfg),
            "vocab_size": vocab.vocab_size,
        },
        path,
    )
    return str(path)


def test_classification_dataset_encodes_text_and_label(vocab):
    dataset = ClassificationDataset(TRAIN_ROWS, vocab, max_len=64)
    item = dataset[1]  # {"text": "we believe and have faith in god", "label": 1}

    assert item["labels"].item() == 1
    assert len(item["input_ids"]) > 0


def test_collate_classification_batch_pads_and_stacks_labels():
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor(0)},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor(2)},
    ]
    out = collate_classification_batch(batch, pad_id=99)

    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [4, 5, 99]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
    assert out["labels"].tolist() == [0, 2]


def test_finetune_classification_runs_end_to_end_and_reports_macro_f1(mlm_checkpoint, vocab, tmp_path):
    result = finetune_classification(
        mlm_checkpoint,
        TRAIN_ROWS,
        EVAL_ROWS,
        vocab=vocab,
        output_dir=str(tmp_path / "out"),
        num_train_epochs=3,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
    )

    assert "eval_macro_f1" in result
    assert 0.0 <= result["eval_macro_f1"] <= 1.0


def test_finetune_classification_with_use_wandb_runs_end_to_end(mlm_checkpoint, vocab, tmp_path, monkeypatch):
    """See test_encoder_finetune_tagging.py's identical test for why
    WANDB_MODE=disabled is safe here (a documented no-op NoopRun, no
    network/login needed) -- confirms use_wandb=True actually flows into
    TrainingArguments(report_to=["wandb"]) and runs cleanly."""
    monkeypatch.setenv("WANDB_MODE", "disabled")

    result = finetune_classification(
        mlm_checkpoint,
        TRAIN_ROWS,
        EVAL_ROWS,
        vocab=vocab,
        output_dir=str(tmp_path / "out_wandb"),
        num_train_epochs=1,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
        use_wandb=True,
        run_name="test-taxi1500-run",
    )

    assert "eval_macro_f1" in result


def test_finetune_classification_with_eval_rows_by_name_evaluates_every_language_without_retraining(mlm_checkpoint, vocab, tmp_path):
    """See test_encoder_finetune_tagging.py's identical test -- one train
    pass, then per-language macro_f1 for every eval_rows_by_name entry via
    Trainer's own eval_dataset=dict[str, Dataset] support."""
    result = finetune_classification(
        mlm_checkpoint,
        TRAIN_ROWS,
        eval_rows=None,
        vocab=vocab,
        output_dir=str(tmp_path / "out_multi"),
        eval_rows_by_name={
            "primary": (EVAL_ROWS, None),
            "extra": (TRAIN_ROWS[:2], None),
        },
        num_train_epochs=1,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
    )

    assert "eval_primary_macro_f1" in result
    assert "eval_extra_macro_f1" in result
    assert "eval_macro_f1" not in result


def test_taxi1500_labels_match_glot500s_own_scheme():
    assert TAXI1500_LABELS == ["Recommendation", "Faith", "Violence", "Grace", "Sin", "Description"]
