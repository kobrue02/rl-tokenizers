"""Boundary/span induction for a trained SuperBPEModel -- mirrors every other
tokenizer package's segment.py shape (a bytes -> list[bytes] spans callable),
even though there's no neural forward pass or discretization step here: BPE's
own encode procedure already produces discrete spans directly (see
superbpe.model.SuperBPEModel.encode_spans), so this is a thin, signature-
matching wrapper, not a real discretization -- kept as its own module (rather
than calling encode_spans directly everywhere) purely so superbpe/cli.py and
superbpe/evaluate.py can import `from .segment import induce_spans` exactly
like every other tokenizer's cli.py/evaluate.py already does, without either
of them needing to know SuperBPE's encoding has no notion of "device" at all.
"""


def induce_spans(model, byte_seq):
    """model: SuperBPEModel. byte_seq: str/bytes. Returns list[bytes] spans."""
    return model.encode_spans(byte_seq)
