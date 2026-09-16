"""Regression test for parity_bpe's TokenizerAdapter support: its
ParityBPEModel is structurally identical to bpe's BPEModel (same
`.tokenizer` attribute over a `tokenizers.Tokenizer`), so it should load
and round-trip through the native family exactly like bpe does."""

import tempfile

from systems.pretraining.tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe
from systems.tokenization.parity_bpe.model import ParityBPEModel

SENTENCES = ["hello world", "hello there", "the quick brown fox", "goodbye world"]


def test_parity_bpe_is_a_native_system():
    assert "parity_bpe" in ALL_SYSTEMS


def test_parity_bpe_adapter_round_trips_like_bpe():
    bpe_model = fit_bpe(SENTENCES, vocab_size=300)
    parity_model = ParityBPEModel(bpe_model.tokenizer)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        checkpoint_path = f.name
    parity_model.tokenizer.save(checkpoint_path)

    adapter = TokenizerAdapter.load("parity_bpe", checkpoint_path)
    ids = adapter.encode("hello world")
    assert adapter.decode(ids) == b"hello world"
