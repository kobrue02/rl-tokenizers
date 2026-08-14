"""Boundary/span induction for a loaded HFFrontierTokenizer -- mirrors every
other tokenizer package's segment.py shape (a bytes -> list[bytes] spans
callable). See model.py's own docstring for why this is a real
discretization step here (unlike bpe/superbpe's thin wrappers): a frontier
tokenizer's own tokens aren't byte spans directly, reconstructing them
correctly is the actual substance of this package.
"""


def induce_spans(wrapped_tokenizer, byte_seq):
    """wrapped_tokenizer: systems.hf_frontier.model.HFFrontierTokenizer.
    byte_seq: str/bytes. Returns list[bytes] spans."""
    return wrapped_tokenizer.induce_spans(byte_seq)
