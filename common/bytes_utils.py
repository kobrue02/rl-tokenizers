"""Generic byte<->tensor and boundary<->span helpers shared by every tokenizer in
this repo (fairtok's RL policy, and the magnet/flexitokens/manta baselines) --
none of this is specific to any one model architecture or training method, just
the common representation every one of them uses: a byte sequence in, a list of
0/1 boundary decisions in, byte spans out.
"""

import torch


def bytes_to_tensor(b, device="cpu"):
    if isinstance(b, str):
        b = b.encode("utf-8")
    return torch.tensor(list(b), dtype=torch.long, device=device)


def spans_from_boundaries(byte_seq, actions):
    """Byte spans induced by boundary decisions (a list of 0/1 ints -- pulled from
    whatever per-position record type a given tokenizer produces, or from a
    deterministic inference-time rollout). Content-keyed by construction (a span IS
    its bytes) -- there is no separate id to dedupe by, which is what makes this
    representation immune to Duplication-BPE-style gaming."""
    if isinstance(byte_seq, torch.Tensor):
        # ONE sync for the whole sequence (a no-op if byte_seq is already CPU, as it
        # is at inference time), not one per span -- an earlier version called
        # `.tolist()` on a fresh tensor slice INSIDE the loop below, so on a CUDA
        # byte_seq (as it is during training) every span forced its own host-device
        # sync. Average span length is ~2-3 bytes, so a ~150-byte sentence produced
        # ~50-75 syncs, times ~150-200 sequences/step -- tens of thousands of
        # syncs/step, the single largest GPU bottleneck found in fairtok's own
        # training loop (see fairtok/train.py's history for the full story).
        byte_seq = byte_seq.detach().tolist()
    spans = []
    start = 0
    last = len(actions) - 1
    for t, action in enumerate(actions):
        if action == 1 or t == last:
            spans.append(bytes(byte_seq[start:t + 1]))
            start = t + 1
    return spans
