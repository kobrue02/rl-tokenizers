"""Shared infrastructure for finetuning systems.pretraining.encoder_train's
MLM encoder on a downstream task -- the piece Glot500's own NER/POS/Taxi1500
protocol needs beyond what encoder_eval.py covers (pseudoperplexity,
retrieval, roundtrip alignment all reuse the MLM checkpoint AS-IS; these
three tasks finetune a NEW task head on top of it, then zero-shot transfer
to other languages, matching Glot500's own run_tag.py/zero_shot_train.py
protocol). This is the one place in systems.pretraining that reaches for
transformers.Trainer directly (per the explicit ask to build this pipeline
on HF's own finetuning infra) rather than a hand-rolled loop like
encoder_train.py/train.py -- appropriate here since AutoModelForTokenClassification/
AutoModelForSequenceClassification + Trainer + a compute_metrics callback
IS the standard way to finetune a classification head in the HF ecosystem,
and there's no project-specific training-loop behavior (FSDP, custom
checkpoint rotation, wandb sample logging, ...) worth reimplementing Trainer
to get, unlike encoder_train.py's own MLM pretraining loop.
"""

import copy

import torch

from .encoder_model import build_encoder
from .encoder_model_configs import get_preset
from .encoder_train import EncoderTrainConfig


def load_finetune_model(checkpoint_path, auto_model_cls, device="cpu", **config_overrides):
    """Loads an encoder_train.py MLM checkpoint and transplants its
    pretrained roberta encoder body into a FRESH auto_model_cls instance
    (AutoModelForTokenClassification/AutoModelForSequenceClassification)
    with a randomly-initialized task head -- confirmed live that both
    classes expose a `.roberta` submodule whose state_dict keys match
    AutoModelForMaskedLM's own exactly, so this is a straight
    load_state_dict, not a manual per-parameter mapping. This sidesteps
    round-tripping through the HF Hub's save_pretrained/from_pretrained
    directory format entirely -- this project's own checkpoints are the
    plain torch.save dict every trainer in this codebase uses (see
    encoder_train.save_checkpoint), not an HF model directory.

    config_overrides: applied to a deepcopy of the MLM model's own config
    before construction (e.g. num_labels=7, id2label={...}, label2id={...})."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EncoderTrainConfig(**ckpt["config"])
    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)

    mlm_model = build_encoder(preset, ckpt["vocab_size"])
    mlm_model.load_state_dict(ckpt["model_state_dict"])

    task_config = copy.deepcopy(mlm_model.config)
    for key, value in config_overrides.items():
        setattr(task_config, key, value)
    task_model = auto_model_cls.from_config(task_config)
    task_model.roberta.load_state_dict(mlm_model.roberta.state_dict())
    return task_model.to(device)
