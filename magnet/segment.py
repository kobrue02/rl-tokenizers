"""Deterministic boundary/span induction for a trained MagnetModel.

Thresholds the boundary probability at 0.5 (no Gumbel-sigmoid sampling) --
matches fairtok.policy.segment_bytes's own `deterministic=True` inference
convention, for the same reason: a frozen model should induce the same
tokenization every time it's asked to segment the same input, not a fresh
stochastic draw each call. See MagnetModel.forward's `sample` flag / model.py's
BoundaryPredictor.forward docstring for how sample=False turns off sampling
inside the model itself.
"""

import torch

from fairtok.policy import bytes_to_tensor, spans_from_boundaries


@torch.no_grad()
def induce_boundaries(model, byte_seq, script, device="cpu"):
    """byte_seq: a 1-D LongTensor of byte ids, or a str/bytes (UTF-8 encoded on
    the fly via fairtok.policy.bytes_to_tensor). Returns a list[int] of 0/1
    boundary decisions, one per byte position -- exactly the format
    fairtok.policy.spans_from_boundaries expects, so it can be passed straight
    through (see induce_spans below, which does exactly that)."""
    if isinstance(byte_seq, (bytes, str)):
        byte_seq = bytes_to_tensor(byte_seq, device)

    was_training = model.training
    model.eval()
    ids = byte_seq.unsqueeze(0).to(device)
    lengths = torch.tensor([ids.shape[1]], device=device)
    _, _, hard_boundaries, _ = model(ids, lengths, script, sample=False)
    model.train(was_training)

    # hard_boundaries is exactly 0.0/1.0 here (sample=False -> no Gumbel noise,
    # and under torch.no_grad() the straight-through "+ soft - soft.detach()"
    # terms cancel exactly) -- .round() is a defensive no-op against floating
    # point noise, not something expected to ever change a value.
    return [int(v) for v in hard_boundaries[0, : lengths[0]].round().tolist()]


def induce_spans(model, byte_seq, script, device="cpu"):
    """Induced boundaries -> byte spans, reusing fairtok.policy.spans_from_boundaries
    unchanged -- this is what makes a MAGNET-tokenized corpus a drop-in input to
    every existing fairtok tool (vocab.py's frequency-table builders,
    metrics.py's compression/fairness metrics) with zero adapter code."""
    if isinstance(byte_seq, (bytes, str)):
        byte_tensor = bytes_to_tensor(byte_seq, device)
    else:
        byte_tensor = byte_seq
    boundaries = induce_boundaries(model, byte_tensor, script, device)
    return spans_from_boundaries(byte_tensor, boundaries)
