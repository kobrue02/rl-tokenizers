"""Generic byte<->tensor and boundary<->span helpers shared by every tokenizer in
this repo -- none of this is specific to any one model architecture or training
method, just the common representation they all use: a byte sequence in, a list
of 0/1 boundary decisions in, byte spans out.
"""

import torch


def bytes_to_tensor(b, device="cpu"):
    if isinstance(b, str):
        b = b.encode("utf-8")
    return torch.tensor(list(b), dtype=torch.long, device=device)


def truncate_to_max_bytes(text, max_bytes):
    """Returns (text, truncated: bool). Truncates on BYTE length, not
    character/codepoint count, since those diverge for multi-byte UTF-8 scripts --
    byte length is what actually drives sequence length T through the neural
    (span-family) systems here. For str input, the truncated byte slice is decoded
    back leniently (errors="ignore") so a multi-byte character split mid-sequence
    is dropped cleanly rather than left as a malformed trailing byte.

    Guards against the O(T^2) memory blowup of manta.model.SlidingWindowAttention's
    dense (B,H,T,T) score matrix -- originally only in fanta training, moved here
    once pretraining.data_prep hit the same blowup at tokenization time (it calls
    encode()/induce_spans on raw corpus text with no length cap of its own)."""
    if not max_bytes:
        return text, False
    if isinstance(text, str):
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text, False
        return encoded[:max_bytes].decode("utf-8", errors="ignore"), True
    if len(text) <= max_bytes:
        return text, False
    return text[:max_bytes], True


def spans_from_boundaries(byte_seq, actions):
    """Byte spans induced by boundary decisions (a list of 0/1 ints). Content-keyed
    by construction (a span IS its bytes) -- no separate id to dedupe by, which is
    what makes this representation immune to Duplication-BPE-style gaming."""
    if isinstance(byte_seq, torch.Tensor):
        # ONE sync for the whole sequence, not one per span: calling `.tolist()`
        # inside the loop below forced a host-device sync per span on a CUDA
        # byte_seq, tens of thousands of syncs/step -- the largest GPU bottleneck
        # found in fairtok's own training loop.
        byte_seq = byte_seq.detach().tolist()
    spans = []
    start = 0
    last = len(actions) - 1
    for t, action in enumerate(actions):
        if action == 1 or t == last:
            spans.append(bytes(byte_seq[start : t + 1]))
            start = t + 1
    return spans
