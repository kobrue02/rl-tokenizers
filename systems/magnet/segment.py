"""Deterministic boundary/span induction for a trained MagnetModel.

Thresholds the boundary probability at 0.5 (no Gumbel-sigmoid sampling) --
matches fairtok.policy.segment_bytes's `deterministic=True` convention, so a
frozen model always induces the same tokenization for the same input. See
BoundaryPredictor.forward for how sample=False disables sampling internally.
"""

import torch

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries


@torch.no_grad()
def induce_boundaries(model, byte_seq, script, device="cpu"):
    """byte_seq: 1-D LongTensor of byte ids, or str/bytes (UTF-8 encoded via
    common.bytes_utils.bytes_to_tensor). Returns a list[int] of 0/1 boundary
    decisions, one per byte position -- the format spans_from_boundaries
    expects."""
    if isinstance(byte_seq, (bytes, str)):
        byte_seq = bytes_to_tensor(byte_seq, device)

    was_training = model.training
    model.eval()
    ids = byte_seq.unsqueeze(0).to(device)
    lengths = torch.tensor([ids.shape[1]], device=device)
    _, _, hard_boundaries, _ = model(ids, lengths, script, sample=False)
    model.train(was_training)

    # hard_boundaries is already exactly 0.0/1.0 (sample=False, no_grad cancels
    # the straight-through terms exactly); .round() is a defensive no-op.
    return [int(v) for v in hard_boundaries[0, : lengths[0]].round().tolist()]


def induce_spans(model, byte_seq, script, device="cpu"):
    """Induced boundaries -> byte spans via spans_from_boundaries, unchanged --
    makes MAGNET output a drop-in input to fairtok's vocab/metrics tooling
    with zero adapter code."""
    if isinstance(byte_seq, (bytes, str)):
        byte_tensor = bytes_to_tensor(byte_seq, device)
    else:
        byte_tensor = byte_seq
    boundaries = induce_boundaries(model, byte_tensor, script, device)
    return spans_from_boundaries(byte_tensor, boundaries)
