"""Boundary/span induction for a trained BPEModel -- mirrors every other
tokenizer package's segment.py shape (a bytes -> list[bytes] spans callable).
See superbpe/segment.py's own docstring for why this is a thin,
signature-matching wrapper rather than a real discretization step: BPE's own
encode procedure already produces discrete spans directly.
"""


def induce_spans(model, byte_seq):
    """model: bpe.model.BPEModel. byte_seq: str/bytes. Returns list[bytes] spans."""
    return model.encode_spans(byte_seq)
