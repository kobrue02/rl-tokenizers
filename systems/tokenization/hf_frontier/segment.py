"""Boundary/span induction for a loaded HFFrontierTokenizer -- mirrors every
other tokenizer package's segment.py shape (bytes -> list[bytes] spans).
Unlike bpe/superbpe's thin wrappers, a frontier tokenizer's tokens aren't
byte spans directly -- reconstructing them (model.py) is the real substance
of this package.
"""


def induce_spans(wrapped_tokenizer, byte_seq):
    """wrapped_tokenizer: systems.tokenization.hf_frontier.model.HFFrontierTokenizer.
    byte_seq: str/bytes. Returns list[bytes] spans."""
    return wrapped_tokenizer.induce_spans(byte_seq)
