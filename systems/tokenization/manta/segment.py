"""Turning MANTa's soft assignment matrix into discrete token boundaries.

MANTa never needs discrete boundaries to train, but common.eval.metrics and
common.vocab expect actual token counts, so something has to collapse the
soft (byte, block) assignment matrix (manta.model.MantaModel) into hard
boundaries for evaluation. This is entirely this project's own modeling
choice, not specified by the paper.

Rule: for each byte i, take `argmax_b P(byte i in block b)`. Since mu_i is a
running cumsum (mu_i <= mu_{i+1}), the argmax block index is monotonically
non-decreasing along the sequence -- so wherever it increases from i to i+1,
that's a boundary.

Convention (common.bytes_utils.spans_from_boundaries): boundary_actions[i]==1
means byte i is the LAST byte of a span, so a block-index increase from i to
i+1 sets boundary_actions[i]=1. The last position needs no explicit 1 --
spans_from_boundaries always closes a span there (matches
fairtok.policy.segment_bytes).

Caveat: early in training the assignment matrix is blurry (frontier probs
near random init, sigma_i large), so the argmax can look near-degenerate --
e.g. everything collapsing to one block, or flickering every byte. Expected
for an undertrained model, not a discretization bug (manta/train.py's smoke
test prints this statistic over training).
"""

import torch

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries


def _to_tensor(byte_seq, device="cpu"):
    """Accepts str/bytes (via common.bytes_utils.bytes_to_tensor) or an
    already-built LongTensor (moved to `device` unchanged) -- the latter lets
    manta.train.py reuse tensors it already holds without a decode/re-encode
    round trip."""
    if torch.is_tensor(byte_seq):
        return byte_seq.to(device)
    return bytes_to_tensor(byte_seq, device)


def boundaries_from_assignment(assignment, lengths):
    """Applies the hard-argmax discretization rule (module docstring) to an
    already-computed assignment matrix (B, T, num_blocks), without a model
    forward pass. Split out from induce_boundaries_batch so train.py's loop
    can reuse the `output.assignment` it already has for the loss, instead of
    a redundant second forward pass to track token-frequency stats.

    Returns: list of 0/1 boundary-action lists, one per batch row, usable by
    common.bytes_utils.spans_from_boundaries.
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
    """Runs the model once over a padded batch and hard-discretizes every
    sequence's assignment matrix into its own boundary list. One forward pass
    per batch instead of per sentence.

    byte_seqs: list of str/bytes/1-D LongTensor (mixed is fine).
    Returns: list of 0/1 boundary-action lists, same order as byte_seqs.
    """
    tensors = [_to_tensor(s, device) for s in byte_seqs]
    lengths = torch.tensor(
        [t.shape[0] for t in tensors], dtype=torch.long, device=device
    )
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
    """Model + one raw byte sequence -> list of byte-string spans, via
    common.bytes_utils.spans_from_boundaries (same span objects fairtok's
    vocab/metrics pipeline consumes)."""
    tensor = _to_tensor(byte_seq, device)
    actions = induce_boundaries(model, tensor, device=device)
    return spans_from_boundaries(tensor, actions)


def induce_spans_batch(model, byte_seqs, device="cpu"):
    """Batched counterpart to induce_spans: one model forward pass for the
    whole list instead of one per document. Same results as calling
    induce_spans per item, far fewer GPU calls.

    Added for systems.pretraining.data_prep, whose previous per-document calls
    (confirmed on a cluster run) badly underutilized the GPU via per-call
    kernel-launch/sync overhead. PADDING CAVEAT: each call pads to the
    longest member of that batch, so memory cost is (batch max length)^2 x
    batch_size -- --max-doc-bytes bounds the worst case, but a larger
    --encode-batch-size scales it up; tune the two together.
    """
    tensors = [_to_tensor(s, device) for s in byte_seqs]
    actions_list = induce_boundaries_batch(model, tensors, device=device)
    return [spans_from_boundaries(t, a) for t, a in zip(tensors, actions_list)]
