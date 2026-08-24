"""Builds an XLM-R-architecture MLM encoder (transformers.AutoModelForMaskedLM
over an XLMRobertaConfig), randomly initialized from scratch -- see
encoder_model_configs.py's own docstring for why FROM SCRATCH rather than
Glot500-m's continued-pretraining recipe.

This is the one file in systems.pretraining that depends on `transformers`
for the model itself -- every other system in this project (model.py,
manta, fairtok, flexitokens, magnet) implements its architecture from
scratch. That trade was made deliberately here: Glot500-m's own architecture
already IS unmodified XLM-R, so reimplementing a bidirectional transformer
encoder from scratch would just be reproducing HF's own, already-correct,
already-tested implementation for no benefit -- unlike the decoder, where
this project's own model.py exists specifically to let --model-size presets
and tokenizer vocab sizes plug together in ways off-the-shelf code doesn't
support identically.
"""

from transformers import AutoModelForMaskedLM, XLMRobertaConfig

from .encoder_tokenizer import PAD_ID


def build_encoder(preset, vocab_size):
    """preset: an encoder_model_configs.EncoderConfig. vocab_size: an
    encoder_tokenizer.EncoderVocab's OWN vocab_size (already includes the
    +2 reserved pad/mask ids -- see encoder_tokenizer's module docstring),
    NOT hardcoded to XLM-R's own 250K/401K -- this is what lets any
    systems/ tokenizer plug in here exactly like it does for the decoder
    (see TokenizerAdapter). pad_token_id is always PAD_ID (0) -- see
    encoder_tokenizer's module docstring for why it must be small and fixed
    rather than a tokenizer-size-dependent large id.

    Returns a randomly-initialized XLMRobertaForMaskedLM
    (AutoModelForMaskedLM.from_config, never .from_pretrained -- see this
    module's own docstring)."""
    hf_cfg = XLMRobertaConfig(
        vocab_size=vocab_size,
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_hidden_layers,
        num_attention_heads=preset.num_attention_heads,
        intermediate_size=preset.intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=preset.hidden_dropout_prob,
        attention_probs_dropout_prob=preset.attention_probs_dropout_prob,
        max_position_embeddings=preset.max_seq_len + 2,  # see EncoderConfig's own docstring
        type_vocab_size=1,  # XLM-R's own convention -- no real segment-type signal used
        layer_norm_eps=preset.layer_norm_eps,
        initializer_range=preset.initializer_range,
        pad_token_id=PAD_ID,
        tie_word_embeddings=preset.tie_word_embeddings,
    )
    return AutoModelForMaskedLM.from_config(hf_cfg)
