"""Deterministic boundary induction for a FROZEN, trained FlexiTokensModel.

Mirrors fairtok.policy.segment_bytes's `deterministic` mode: threshold the boundary
probability at 0.5, no Gumbel-sigmoid sampling (that noise only exists during
training, to give the straight-through estimator something stochastic to explore --
see flexitokens/model.py's module docstring, point 3). Producing a plain list of 0/1
ints per byte position means a trained FlexiTokensModel's output plugs directly into
common.bytes_utils.spans_from_boundaries for span/vocabulary extraction, exactly the same
as a trained fairtok.policy.BytePolicy checkpoint does -- the two tokenizers are
interchangeable from that point in the pipeline onward.
"""

import torch

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries

from .model import (
    FlexiTokensModel,
)  # noqa: F401  (re-exported for convenience / type hints)


@torch.no_grad()
def induce_boundaries(model, byte_seq, device="cpu"):
    """byte_seq: str, bytes, or a 1-D LongTensor of byte ids -- anything
    common.bytes_utils.bytes_to_tensor already accepts. Returns a plain list of 0/1
    ints, one per byte position, length == len(byte_seq). Deterministic: same
    model + same input always gives the same output (unlike training, which
    samples from the Gumbel-sigmoid relaxation)."""
    if not isinstance(byte_seq, torch.Tensor):
        byte_seq = bytes_to_tensor(byte_seq, device)
    byte_seq = byte_seq.to(device)

    was_training = model.training
    model.eval()
    batch = byte_seq.unsqueeze(0)  # (1, T) -- single-sequence "batch"
    lengths = torch.tensor([batch.shape[1]], device=device)
    out = model(batch, lengths, deterministic=True)
    model.train(was_training)

    boundaries = out["boundaries"][0, : byte_seq.shape[0]]
    return boundaries.round().long().cpu().tolist()


def induce_spans(model, byte_seq, device="cpu"):
    """Convenience wrapper: induce_boundaries + common.bytes_utils.spans_from_boundaries
    -- the same two-step pipeline fairtok.policy.segment_bytes does internally for
    BytePolicy checkpoints, just with this module's model/boundary source instead."""
    byte_seq_t = (
        byte_seq
        if isinstance(byte_seq, torch.Tensor)
        else bytes_to_tensor(byte_seq, device)
    )
    boundaries = induce_boundaries(model, byte_seq_t, device)
    return spans_from_boundaries(byte_seq_t, boundaries)
