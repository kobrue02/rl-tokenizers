"""Turning MANTa's soft assignment matrix into discrete token boundaries.

MANTa itself never needs discrete boundaries -- the paper's whole point is
that the soft (byte, block) assignment matrix (see manta.model.MantaModel)
is enough to train an end-to-end language model without ever materializing
a hard segmentation. But common.metrics (compression_rate, renyi_efficiency,
gini_coefficient) and common.vocab all expect actual token COUNTS -- there
is no way to plug a soft assignment matrix into "how many tokens did this
sentence take," so *something* has to collapse it to hard boundaries for
evaluation. This module is that something, and it is entirely this
project's own modeling choice, not something the MANTa paper specifies.

The rule (as given in the task spec): for each byte position i, take
`argmax_b P(byte i in block b)` -- the single most likely block, per the
assignment matrix's own softmax. Block indices are monotonically
non-decreasing along the sequence by construction (mu_i is a running
cumulative sum, so mu_i <= mu_{i+1} for all i, and the whole Gaussian family
shifts right as i increases) -- so wherever the argmax block index INCREASES
from position i to i+1, that's a boundary: byte i is the last byte of its
block, byte i+1 starts a new one.

Encoded in fairtok's own convention (common.bytes_utils.spans_from_boundaries):
boundary_actions[i] == 1 means "byte i is the LAST byte of a span" -- so a
block-index increase from i to i+1 sets boundary_actions[i] = 1, not
boundary_actions[i+1]. The very last position doesn't need an explicit 1:
spans_from_boundaries always closes a span there regardless of its own
action value, matching fairtok.policy.segment_bytes's own convention.

Caveat surfaced deliberately, not hidden: early in training the soft
assignment matrix is blurry (frontier probabilities near their random
initialization, sigma_i large relative to the spacing between candidate
blocks), so the argmax can be noisy or nearly degenerate -- e.g. almost
every byte argmaxing to the same one or two blocks (near full-sentence
collapse) or the argmax flickering block-to-block almost every byte
(near single-byte collapse). That is EXPECTED behavior for an undertrained
model, not a bug in this discretization -- see manta/train.py's smoke test,
which prints exactly this statistic over the course of training so the
effect is visible rather than papered over.
"""

import torch

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries


def _to_tensor(byte_seq, device="cpu"):
    """Accepts str/bytes (delegates to common.bytes_utils.bytes_to_tensor, so the
    byte<->tensor convention is identical to fairtok's own) or an
    already-built LongTensor (just moved to `device`, unchanged) -- the
    latter matters for manta.train.py, which already holds padded tensors
    it wants to reuse without a decode/re-encode round trip."""
    if torch.is_tensor(byte_seq):
        return byte_seq.to(device)
    return bytes_to_tensor(byte_seq, device)


def boundaries_from_assignment(assignment, lengths):
    """Lower-level primitive: given an ALREADY-COMPUTED assignment matrix
    (B, T, num_blocks) and each row's real length, apply the hard-argmax
    discretization rule (see module docstring) without running the model
    again. Split out from induce_boundaries_batch so manta.train.py's
    training loop -- which already has `output.assignment` sitting around
    from the forward pass it needed for the loss anyway -- can reuse it
    for free instead of paying for a second, redundant forward pass just
    to track token-frequency statistics during training.

    Returns: list of 0/1 boundary-action lists, one per batch row, each
    directly usable by common.bytes_utils.spans_from_boundaries.
    """
    block_idx = assignment.argmax(dim=-1)  # (B, T)
    results = []
    for i in range(assignment.shape[0]):
        L = int(lengths[i].item())
        idx = block_idx[i, :L].tolist()
        actions = [0] * L
        for t in range(1, L):
            if idx[t] > idx[t - 1]:
                actions[t - 1] = 1  # byte t-1 is the last byte of its block
        results.append(actions)
    return results


@torch.no_grad()
def induce_boundaries_batch(model, byte_seqs, device="cpu"):
    """Run the model once over a padded batch and hard-discretize every
    sequence's assignment matrix into its own boundary list (see module
    docstring for the exact rule). Batching this (rather than calling the
    single-sequence version in a Python loop) means building a vocabulary
    or scoring many sentences at evaluation time costs one forward pass per
    batch, not one per sentence.

    byte_seqs: list of str/bytes/1-D LongTensor (mixed is fine).
    Returns: list of 0/1 boundary-action lists, same order/length as
    byte_seqs, each directly usable by common.bytes_utils.spans_from_boundaries.
    """
    tensors = [_to_tensor(s, device) for s in byte_seqs]
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long, device=device)
    T = int(lengths.max().item()) if len(tensors) else 0
    B = len(tensors)
    padded = torch.zeros(B, T, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t

    was_training = model.training
    model.eval()
    output = model(padded, lengths)
    model.train(was_training)
    return boundaries_from_assignment(output.assignment, lengths)


def induce_boundaries(model, byte_seq, device="cpu"):
    """Single-sequence convenience wrapper around induce_boundaries_batch."""
    return induce_boundaries_batch(model, [byte_seq], device=device)[0]


def induce_spans(model, byte_seq, device="cpu"):
    """Model + one raw byte sequence -> a list of byte-string spans, via
    common.bytes_utils.spans_from_boundaries (reused unmodified, so the spans
    this produces are byte-for-byte the same kind of object fairtok's own
    vocab/metrics pipeline already consumes)."""
    tensor = _to_tensor(byte_seq, device)
    actions = induce_boundaries(model, tensor, device=device)
    return spans_from_boundaries(tensor, actions)
