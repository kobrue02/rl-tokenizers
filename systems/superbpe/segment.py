"""Boundary/span induction for a trained SuperBPEModel -- mirrors every other
tokenizer's segment.py shape (bytes -> list[bytes] spans), though there's no
real discretization here: BPE's encode already produces discrete spans
directly (SuperBPEModel.encode_spans). Kept as its own module so cli.py/
evaluate.py can import `from .segment import induce_spans` like every other
tokenizer, without needing to know SuperBPE has no "device" concept.
"""


def induce_spans(model, byte_seq):
    """model: SuperBPEModel. byte_seq: str/bytes. Returns list[bytes] spans."""
    return model.encode_spans(byte_seq)
