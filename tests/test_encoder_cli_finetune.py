"""Smoke tests for encoder_cli_finetune.py's dispatch wiring: --task ner/pos
against a monkeypatched datasets.load_dataset (no real network dependency),
and --task taxi1500 against a monkeypatched download/parse pair -- same
convention as test_encoder_cli_eval.py's own monkeypatched stream_groups tests."""

import dataclasses

import pytest
import torch

import systems.pretraining.encoder_cli_finetune as cli_finetune_module
from systems.pretraining.encoder_cli_finetune import UPOS_LABELS, WIKIANN_LABELS, main, run_ner, run_pos, run_taxi1500
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_tokenizer import EncoderVocab
from systems.pretraining.encoder_train import EncoderTrainConfig
from systems.pretraining.tokenizer_adapter import TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe


class _FakeHFDataset:
    """Minimal stand-in for a datasets.Dataset: len(), integer indexing
    returning a dict row, and .select(range(...)) -- everything
    encoder_cli_finetune._capped/TaggingDataset actually use."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def select(self, indices):
        return _FakeHFDataset([self.rows[i] for i in indices])

    def __iter__(self):
        return iter(self.rows)


NER_ROWS = {
    "en": [{"tokens": ["John", "works", "at", "Acme"], "ner_tags": [1, 0, 0, 3]}] * 4,
    "de": [{"tokens": ["Hans", "arbeitet", "bei", "Acme"], "ner_tags": [1, 0, 0, 3]}] * 2,
}
POS_ROWS = {
    "en_ewt": [{"tokens": ["the", "dog", "runs"], "upos": ["DET", "NOUN", "VERB"]}] * 4,
    "de_gsd": [{"tokens": ["der", "Hund", "läuft"], "upos": ["DET", "NOUN", "VERB"]}] * 2,
}


@pytest.fixture
def bpe_checkpoint(tmp_path):
    sentences = [
        "John works at Acme Hans arbeitet bei",
        "the dog runs der Hund lauft",
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
            "step": 10, "model_state_dict": model.state_dict(), "optimizer_state_dict": {},
            "scheduler_state_dict": {}, "config": dataclasses.asdict(cfg), "vocab_size": vocab.vocab_size,
        },
        path,
    )
    return str(path)


class _Args:
    def __init__(self, **kwargs):
        self.use_wandb = False  # default -- overridable via kwargs, matches
        # build_arg_parser's own --use-wandb default of False
        self.run_name = ""
        self.__dict__.update(kwargs)


def test_run_ner_dispatches_wikiann_and_reports_f1(mlm_checkpoint, vocab, tmp_path, monkeypatch, capsys):
    def fake_load_dataset(name, config, split):
        assert name == "unimelb-nlp/wikiann"
        return _FakeHFDataset(NER_ROWS[config])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    args = _Args(
        checkpoint=mlm_checkpoint, train_lang="en", eval_lang="de",
        max_train_examples=None, max_eval_examples=None,
        output_dir=str(tmp_path / "out"), device="cpu",
    )
    run_ner(args, vocab)

    out = capsys.readouterr().out
    assert "ner: 4 train (en) / 2 eval (de) rows" in out
    assert "eval_f1=" in out


def test_run_pos_dispatches_universal_dependencies_and_reports_f1(mlm_checkpoint, vocab, tmp_path, monkeypatch, capsys):
    def fake_load_dataset(name, config, split):
        assert name == "universal-dependencies/universal_dependencies"
        return _FakeHFDataset(POS_ROWS[config])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    args = _Args(
        checkpoint=mlm_checkpoint, train_config="en_ewt", eval_config="de_gsd",
        max_train_examples=None, max_eval_examples=None,
        output_dir=str(tmp_path / "out"), device="cpu",
    )
    run_pos(args, vocab)

    out = capsys.readouterr().out
    assert "pos: 4 train (en_ewt) / 2 eval (de_gsd) rows" in out
    assert "eval_accuracy=" in out


def test_run_taxi1500_dispatches_download_and_reports_macro_f1(mlm_checkpoint, vocab, tmp_path, monkeypatch, capsys):
    rows_by_split = {
        "train": "1\tFaith\twe believe\n2\tSin\the sinned\n3\tGrace\tgrace given\n4\tRecommendation\tdo this\n",
        "test": "5\tFaith\tour faith\n6\tSin\tsin again\n",
    }

    def fake_download(split, cache_dir):
        path = tmp_path / f"eng_{split}.tsv"
        path.write_text(rows_by_split[split])
        return str(path)

    monkeypatch.setattr(cli_finetune_module, "download_taxi1500_split", fake_download)

    args = _Args(
        checkpoint=mlm_checkpoint, taxi1500_cache_dir=str(tmp_path), taxi1500_eval_tsv="",
        max_train_examples=None, max_eval_examples=None,
        output_dir=str(tmp_path / "out"), device="cpu",
    )
    run_taxi1500(args, vocab)

    out = capsys.readouterr().out
    assert "taxi1500: 4 train (eng) / 2 eval" in out
    assert "eval_macro_f1=" in out


def test_main_with_use_wandb_runs_end_to_end_for_taxi1500(mlm_checkpoint, bpe_checkpoint, tmp_path, monkeypatch, capsys):
    """--use-wandb must flow all the way through main() -> run_taxi1500 ->
    finetune_classification's own use_wandb=True path. WANDB_MODE=disabled
    makes wandb.init() a documented no-op (NoopRun, no network/login
    needed) -- confirmed live, see test_encoder_finetune_classification.py's
    identical reasoning."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    rows_by_split = {
        "train": "1\tFaith\twe believe\n2\tSin\the sinned\n3\tGrace\tgrace given\n4\tRecommendation\tdo this\n",
        "test": "5\tFaith\tour faith\n6\tSin\tsin again\n",
    }

    def fake_download(split, cache_dir):
        path = tmp_path / f"eng_{split}.tsv"
        path.write_text(rows_by_split[split])
        return str(path)

    monkeypatch.setattr(cli_finetune_module, "download_taxi1500_split", fake_download)

    main([
        "--checkpoint", mlm_checkpoint,
        "--system", "bpe", "--tokenizer-checkpoint", bpe_checkpoint,
        "--task", "taxi1500",
        "--taxi1500-cache-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--device", "cpu",
        "--use-wandb", "--run-name", "test-cli-run",
    ])

    out = capsys.readouterr().out
    assert "logged finetune results to wandb project=" in out


def test_wikiann_and_upos_label_lists_are_fixed_and_correctly_ordered():
    assert WIKIANN_LABELS == ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    assert len(UPOS_LABELS) == 17
    assert len(set(UPOS_LABELS)) == 17
