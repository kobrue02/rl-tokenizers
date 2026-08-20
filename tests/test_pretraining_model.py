"""Tests for systems.pretraining.model's vocab-size padding (a tensor-core
throughput fix matching lit-llama's find_multiple(vocab_size, 64)
convention -- see _padded_vocab_size's own docstring) and the resulting
generate()-time masking that keeps sampling from ever returning a padded,
invalid token id."""

import torch

from systems.pretraining.model import TransformerLM, _padded_vocab_size
from systems.pretraining.model_configs import get_preset


def test_padded_vocab_size_already_a_multiple_is_unchanged():
    assert _padded_vocab_size(64) == 64
    assert _padded_vocab_size(128) == 128


def test_padded_vocab_size_rounds_up_to_next_multiple():
    assert _padded_vocab_size(50000) == 50048  # 50000 -> next multiple of 64
    assert _padded_vocab_size(1) == 64
    assert _padded_vocab_size(65) == 128


def test_embed_and_lm_head_use_padded_size_not_true_vocab_size():
    true_vocab_size = 1000  # deliberately not a multiple of 64
    model = TransformerLM(get_preset("tiny"), true_vocab_size)

    assert model.vocab_size == true_vocab_size  # true size still tracked
    assert model.embed.weight.shape[0] == _padded_vocab_size(true_vocab_size)
    assert model.lm_head.weight.shape[0] == _padded_vocab_size(true_vocab_size)
    assert model.embed.weight.shape[0] != true_vocab_size  # actually padded


def test_forward_and_loss_work_with_labels_below_true_vocab_size():
    true_vocab_size = 100
    model = TransformerLM(get_preset("tiny"), true_vocab_size)
    x = torch.randint(0, true_vocab_size, (2, 16))
    y = torch.randint(0, true_vocab_size, (2, 16))

    logits, loss = model(x, labels=y)

    assert logits.shape == (2, 16, _padded_vocab_size(true_vocab_size))
    assert torch.isfinite(loss)


def test_generate_never_returns_a_padded_invalid_token_id():
    true_vocab_size = 100  # padded to 128 -- 28 invalid slots to guard against
    model = TransformerLM(get_preset("tiny"), true_vocab_size)
    prompt = torch.randint(0, true_vocab_size, (1, 4))

    out = model.generate(prompt, max_new_tokens=40, temperature=2.0)  # high temp -- more likely to expose unmasked slots

    assert out.max().item() < true_vocab_size
    assert out.min().item() >= 0
