"""Thin layer over TokenizerAdapter adding the two special ids an MLM
encoder needs that the decoder pipeline never does: a mask token (for
masked-position substitution) and a pad token (for batching variable-length
real sentences at eval time -- training windows are always fixed-length, see
encoder_data.py, so padding never comes up there). TokenizerAdapter itself
is untouched: every system/checkpoint combination the decoder pipeline
already supports works here unchanged.

PAD_ID/MASK_ID are fixed at 0/1, with every REAL token id shifted up by
RESERVED (2) -- NOT appended after the real vocabulary the way
TokenizerAdapter's own eos_id is. This is forced by an HF implementation
detail, not a style choice: XLMRobertaModel's position embeddings reuse
config.pad_token_id as the position-embedding table's OWN padding_idx too
(RoBERTa's create_position_ids_from_input_ids convention offsets real
position ids by padding_idx), and nn.Embedding asserts padding_idx <
num_embeddings at construction time -- so pad_token_id must be small
(comfortably under max_position_embeddings, ~512) and FIXED, not a
tokenizer-size-dependent large id like eos_id is. Confirmed live: appending
pad/mask after a several-thousand-token vocab crashes
XLMRobertaModel.__init__ with "Padding_idx must be within num_embeddings"
on the position embedding table specifically.
"""

import dataclasses

from .tokenizer_adapter import TokenizerAdapter

PAD_ID = 0
MASK_ID = 1
RESERVED = 2  # count of ids reserved below every real token id -- see module docstring


@dataclasses.dataclass
class EncoderVocab:
    adapter: TokenizerAdapter

    def __post_init__(self):
        self.pad_id = PAD_ID
        self.mask_id = MASK_ID
        self.vocab_size = self.adapter.vocab_size + RESERVED

    @classmethod
    def load(cls, system, checkpoint_path, vocab_json_path=None, device="cpu"):
        return cls(TokenizerAdapter.load(system, checkpoint_path, vocab_json_path, device=device))

    def encode(self, text, lang=None):
        return [i + RESERVED for i in self.adapter.encode(text, lang=lang)]

    def decode(self, ids):
        return self.adapter.decode([i - RESERVED for i in ids if i >= RESERVED])
