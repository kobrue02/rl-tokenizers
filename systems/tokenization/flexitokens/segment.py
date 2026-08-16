"""Deterministic boundary induction for a frozen, trained FlexiTokensModel.

Mirrors fairtok.policy.segment_bytes's `deterministic` mode: thresholds the
boundary probability at 0.5, no Gumbel-sigmoid sampling (that noise only
exists during training, for the straight-through estimator to explore). A
plain list of 0/1 ints per byte position plugs directly into
common.bytes_utils.spans_from_boundaries, same as fairtok.policy.BytePolicy --
the two tokenizers are interchangeable from that point onward.
"""

import torch

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries


@torch.no_grad()
def induce_boundaries(model, byte_seq, device="cpu"):
    """byte_seq: str, bytes, or 1-D LongTensor -- anything bytes_to_tensor
    accepts. Returns a plain list of 0/1 ints, one per byte position.
    Deterministic: same model + input always gives the same output (unlike
    training's Gumbel-sigmoid sampling)."""
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
    """Convenience wrapper: induce_boundaries + spans_from_boundaries -- same
    two-step pipeline fairtok.policy.segment_bytes does internally, with this
    module's model as the boundary source."""
    byte_seq_t = (
        byte_seq
        if isinstance(byte_seq, torch.Tensor)
        else bytes_to_tensor(byte_seq, device)
    )
    boundaries = induce_boundaries(model, byte_seq_t, device)
    return spans_from_boundaries(byte_seq_t, boundaries)
