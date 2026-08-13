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


def truncate_to_max_bytes(text, max_bytes):
    """Returns (text, truncated: bool). Truncates on BYTE length, not
    character/codepoint count, since those diverge for any multi-byte
    UTF-8 script (the exact languages this project cares most about not
    shortchanging) -- what actually drives sequence length T through any
    of the neural (span-family) systems here. For str input, the truncated
    byte slice is decoded back leniently (errors="ignore") so a multi-byte
    character split mid-sequence is dropped cleanly rather than left as a
    malformed trailing byte.

    Originally lived only in systems.fanta.train (added there for
    FantaConfig.max_seq_length, guarding manta.model.SlidingWindowAttention's
    dense (B,H,T,T) score matrix during TRAINING) -- moved here once
    pretraining.data_prep hit the exact same O(T^2) memory blowup from the
    exact same attention mechanism, but at TOKENIZATION time: data_prep
    calls a system's encode()/induce_spans directly on whatever raw corpus
    document text comes through, with no length cap of its own, so any
    unusually long document hits this regardless of what a system's own
    training-time config capped sequences to. One shared truncation
    primitive, used at both call sites, rather than two copies that could
    drift apart."""
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
            spans.append(bytes(byte_seq[start : t + 1]))
            start = t + 1
    return spans
