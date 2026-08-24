"""Tests for systems.pretraining.encoder_model/encoder_model_configs/
encoder_tokenizer: the from-scratch-initialized XLM-R-architecture MLM
encoder (see encoder_model.py's own docstring for why AutoModelForMaskedLM.
from_config, never .from_pretrained)."""

import torch

from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_tokenizer import PAD_ID


def test_build_encoder_is_randomly_initialized_not_pretrained():
    """.from_config (not .from_pretrained) must never reach the network or
    load real XLM-R weights -- confirmed indirectly by asserting the
    embedding table is sized to OUR vocab_size, not XLM-R's own 250K, which
    .from_pretrained("xlm-roberta-base") would otherwise produce regardless
    of the vocab_size argument."""
    preset = get_preset("tiny")
    model = build_encoder(preset, vocab_size=100)

    assert model.config.vocab_size == 100
    assert model.config.pad_token_id == PAD_ID
    assert model.get_input_embeddings().weight.shape == (100, preset.hidden_size)


def test_build_encoder_ties_embeddings_by_default():
    preset = get_preset("tiny")
    model = build_encoder(preset, vocab_size=100)

    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def test_pad_token_id_is_fixed_and_small_regardless_of_vocab_size():
    """Real bug this guards against: HF's RobertaEmbeddings reuses
    config.pad_token_id as the POSITION-embedding table's own padding_idx
    too, which asserts padding_idx < max_position_embeddings at
    construction time. Appending pad_id after a large real vocabulary (the
    natural-looking choice, mirroring TokenizerAdapter's own eos_id
    convention) crashes model construction the moment vocab_size exceeds
    max_position_embeddings -- confirmed live. PAD_ID=0 must stay small and
    fixed however large vocab_size is."""
    preset = get_preset("tiny")  # max_seq_len=32 -> max_position_embeddings=34
    model = build_encoder(preset, vocab_size=50_000)  # far past 34
    assert model.config.pad_token_id == PAD_ID == 0


def test_forward_with_labels_only_scores_masked_positions():
    """HF's own MLM loss (CrossEntropyLoss(ignore_index=-100)) should match
    a manual masked-position cross-entropy computed from the same logits --
    this is really a confirmation that transformers' own convention lines
    up with what encoder_data.mlm_collate_fn produces (labels=-100 outside
    the sampled mask), not a test of transformers' internals."""
    preset = get_preset("tiny")
    vocab_size = 50
    model = build_encoder(preset, vocab_size=vocab_size)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (2, 10))
    labels = torch.full((2, 10), -100)
    labels[0, 3] = input_ids[0, 3].item()
    labels[1, 7] = input_ids[1, 7].item()

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        logits = model(input_ids=input_ids).logits

    manual_loss = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 3], logits[1, 7]]),
        torch.stack([labels[0, 3], labels[1, 7]]),
    )
    assert torch.allclose(out.loss, manual_loss, atol=1e-5)


def test_bidirectional_attention_a_masked_token_sees_future_context():
    """Unlike the decoder's causal Attention (model.py), this encoder must
    NOT restrict attention to a causal prefix -- confirmed by checking that
    changing a token AFTER a given position changes that position's own
    output logits (impossible under causal masking)."""
    preset = get_preset("tiny")
    vocab_size = 50
    model = build_encoder(preset, vocab_size=vocab_size)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, 10))
    with torch.no_grad():
        logits_a = model(input_ids=input_ids).logits[0, 2]

    changed = input_ids.clone()
    changed[0, 8] = (changed[0, 8] + 1) % vocab_size
    with torch.no_grad():
        logits_b = model(input_ids=changed).logits[0, 2]

    assert not torch.allclose(logits_a, logits_b)
