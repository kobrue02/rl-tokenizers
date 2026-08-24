"""Named size presets for the MLM encoder (systems.pretraining.encoder_model),
mirroring model_configs.py's PRESETS convention for the decoder -- "base"/
"large" reproduce XLM-R-base/-large's own published architecture dimensions
exactly (hidden_size/num_hidden_layers/num_attention_heads/intermediate_size),
the same dimensions Glot500-m itself uses: Glot500-m IS continued-pretrained
XLM-R-base with an unchanged transformer body (its paper's Table 2 lists
"Transformer Size: 86M" for both XLM-R-base and Glot500-m -- only the
embedding table grows, from vocabulary extension).

UNLIKE Glot500-m, encoder_model.build_encoder trains FROM SCRATCH
(transformers.AutoModelForMaskedLM.from_config(...), never .from_pretrained(...))
with each systems/ tokenizer's own vocabulary -- consistent with every other
pretraining run in this project (systems.pretraining.model's decoder is
always randomly initialized too, see TrainConfig's docstring). Continued-
pretraining actual XLM-R weights would tie most parameters to XLM-R's OWN
250K-token SentencePiece vocabulary's already-learned semantics, which isn't
a fair "same architecture, different tokenizer" comparison point against
this project's other from-scratch tokenizer experiments.
"""

import dataclasses


@dataclasses.dataclass
class EncoderConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_seq_len: int = 512  # encoder_model.build_encoder sets
    # max_position_embeddings = max_seq_len + 2 -- RoBERTa/XLM-R's own
    # padding_idx=1 convention offsets real position ids to start at 2, so
    # the position embedding table needs 2 extra rows beyond the true
    # sequence length (see RoBERTa's create_position_ids_from_input_ids).
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True  # RoBERTa/XLM-R convention: MLM head's
    # decoder weight IS the input embedding, same as this project's decoder
    # ModelConfig.tie_embeddings default.


PRESETS = {
    "tiny": EncoderConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, max_seq_len=32,
    ),  # smoke testing only -- matches model_configs.PRESETS["tiny"]'s own role for the decoder
    "base": EncoderConfig(),  # XLM-R-base dims -- ~86M transformer-body
    # params, matching Glot500-m's own (see module docstring)
    "large": EncoderConfig(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
    ),  # XLM-R-large dims -- ~303M transformer-body params
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown encoder size {name!r} -- choose from {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])  # copy so callers mutating
    # their instance (e.g. overriding max_seq_len) don't mutate the shared preset
