"""Boundary/span induction for a trained BPEModel -- mirrors every other
tokenizer package's segment.py shape (bytes -> list[bytes] spans). Thin,
signature-matching wrapper rather than a real discretization step: BPE's
encode already produces discrete spans directly (see superbpe/segment.py).
"""


def induce_spans(model, byte_seq):
    """model: bpe.model.BPEModel. byte_seq: str/bytes. Returns list[bytes] spans."""
    return model.encode_spans(byte_seq)
