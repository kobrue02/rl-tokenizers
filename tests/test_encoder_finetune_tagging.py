"""End-to-end test for encoder_finetune_tagging: real Trainer.train() +
Trainer.evaluate() over a tiny synthetic in-memory NER-shaped dataset (no
real WikiANN/network dependency -- a caller is expected to pass in
whatever datasets.load_dataset(...) split they built themselves; this
module has no HF Hub dependency of its own, see its module docstring)."""

import dataclasses

import pytest
import torch

import numpy as np

from systems.pretraining.encoder_finetune_tagging import (
    TaggingDataset,
    build_compute_metrics,
    collate_tagging_batch,
    finetune_tagging,
)
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_tokenizer import PAD_ID, EncoderVocab
from systems.pretraining.encoder_train import EncoderTrainConfig
from systems.pretraining.tokenizer_adapter import TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

# A tiny hand-built NER-shaped corpus -- "tokens"/"ner_tags" is exactly HF
# datasets' own WikiANN row shape.
TRAIN_ROWS = [
    {"tokens": ["John", "works", "at", "Acme"], "ner_tags": [1, 0, 0, 3]},
    {"tokens": ["Paris", "is", "nice"], "ner_tags": [5, 0, 0]},
    {"tokens": ["Mary", "visited", "Berlin"], "ner_tags": [1, 0, 5]},
    {"tokens": ["Acme", "hired", "John"], "ner_tags": [3, 0, 1]},
]
EVAL_ROWS = [
    {"tokens": ["Peter", "lives", "in", "London"], "ner_tags": [1, 0, 0, 5]},
]


@pytest.fixture
def bpe_checkpoint(tmp_path):
    sentences = [
        "John works at Acme in Paris and Berlin",
        "Mary visited London and hired Peter",
        "a small tokenizer trained only for this test",
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


def test_tagging_dataset_aligns_first_subword_and_ignores_the_rest(vocab):
    dataset = TaggingDataset(TRAIN_ROWS, "ner_tags", vocab, max_len=64)
    item = dataset[0]  # {"tokens": ["John", "works", "at", "Acme"], "ner_tags": [1, 0, 0, 3]}

    # Every real label (1, 0, 0, 3) must appear in the label sequence
    # exactly once each (at each word's first subword), everything else -100.
    labels = item["labels"].tolist()
    non_ignored = [l for l in labels if l != -100]
    assert non_ignored == [1, 0, 0, 3]
    assert len(item["input_ids"]) == len(labels)


def test_tagging_dataset_truncates_to_max_len(vocab):
    dataset = TaggingDataset(TRAIN_ROWS, "ner_tags", vocab, max_len=2)
    item = dataset[0]
    assert len(item["input_ids"]) == 2
    assert len(item["labels"]) == 2


def test_collate_tagging_batch_pads_and_builds_attention_mask():
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([0, -100, 1])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, 2])},
    ]
    out = collate_tagging_batch(batch, pad_id=99)

    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [4, 5, 99]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
    assert out["labels"][1].tolist() == [-100, 2, -100]


def test_build_compute_metrics_bio_scores_entity_level_f1():
    label_list = LABEL_LIST
    compute_metrics = build_compute_metrics(label_list, scheme="bio")
    # One row, 4 tokens: predicts B-PER correctly, misses the ORG entirely (predicts O).
    labels = np.array([[1, 0, 0, 3]])  # B-PER, O, O, B-ORG
    logits = np.zeros((1, 4, len(label_list)))
    predicted = [1, 0, 0, 0]  # last one wrong (predicts O instead of B-ORG)
    for pos, lab in enumerate(predicted):
        logits[0, pos, lab] = 10.0

    result = compute_metrics((logits, labels))

    assert set(result.keys()) == {"precision", "recall", "f1"}
    assert 0.0 < result["recall"] < 1.0  # found 1 of 2 entities


def test_build_compute_metrics_flat_scores_per_token_accuracy_not_entity_f1():
    """The whole reason POS needs scheme="flat" -- see build_compute_metrics'
    own docstring: seqeval's entity scorer misbehaves on non-BIO tags like
    UPOS's ADJ/NOUN/VERB. Confirms the "flat" path returns a plain
    per-token accuracy instead."""
    label_list = ["DET", "NOUN", "VERB"]
    compute_metrics = build_compute_metrics(label_list, scheme="flat")
    labels = np.array([[0, 1, 2, -100]])  # DET, NOUN, VERB, padding
    logits = np.zeros((1, 4, len(label_list)))
    predicted = [0, 1, 0, 0]  # 2 of 3 real positions correct
    for pos, lab in enumerate(predicted):
        logits[0, pos, lab] = 10.0

    result = compute_metrics((logits, labels))

    assert result == {"accuracy": pytest.approx(2 / 3)}


def test_build_compute_metrics_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown scheme"):
        build_compute_metrics(LABEL_LIST, scheme="bogus")


def test_finetune_tagging_runs_end_to_end_and_reports_seqeval_metrics(mlm_checkpoint, vocab, tmp_path):
    result = finetune_tagging(
        mlm_checkpoint,
        TRAIN_ROWS,
        EVAL_ROWS,
        tag_column="ner_tags",
        label_list=LABEL_LIST,
        vocab=vocab,
        output_dir=str(tmp_path / "out"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
    )

    assert "eval_f1" in result
    assert "eval_precision" in result
    assert "eval_recall" in result
    assert 0.0 <= result["eval_f1"] <= 1.0


def test_finetune_tagging_with_use_wandb_runs_end_to_end(mlm_checkpoint, vocab, tmp_path, monkeypatch):
    """use_wandb=True must actually flow into TrainingArguments(report_to=
    ["wandb"]) and run cleanly through a real Trainer.train() -- WANDB_MODE
    =disabled makes wandb.init() a documented no-op (NoopRun, no network/
    login), so this exercises the real code path without needing a real
    wandb account (see this project's own research before writing this:
    confirmed live that WANDB_MODE=disabled returns a NoopRun)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")

    result = finetune_tagging(
        mlm_checkpoint,
        TRAIN_ROWS,
        EVAL_ROWS,
        tag_column="ner_tags",
        label_list=LABEL_LIST,
        vocab=vocab,
        output_dir=str(tmp_path / "out_wandb"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
        use_wandb=True,
        run_name="test-ner-run",
    )

    assert "eval_f1" in result
