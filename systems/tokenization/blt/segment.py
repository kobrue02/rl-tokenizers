"""Turning the entropy model's per-byte entropy into BLT's actual dynamic
patch boundaries (Pagnoni et al. 2024's `patching_mode="entropy"`,
`monotonicity=false` rule -- see model.py's module docstring for where this
config value actually lives and the entropy_model/params.json trap).

Traced verbatim from bytelatent/data/patcher.py's find_entropy_patch_start_ids
(monotonicity=False, threshold_add=None branch):

  - token index 0 (BOS) and token index 1 (the first real byte) are ALWAYS
    patch starts -- there's no prior context to judge either by.
  - for every token index j >= 2: j starts a new patch iff
    entropies[j-1] > PATCHING_THRESHOLD.

Note the off-by-one, easy to get backwards: it's the entropy of PREDICTING
token j (i.e. the model's uncertainty right before emitting it, computed
from tokens[0..j-1]) that gates whether j opens a new patch -- not the
entropy of predicting whatever comes after j.
"""

import torch
import torch.nn.functional as F

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries

from .model import BOS_ID, EOS_ID, OFFSET, PATCHING_THRESHOLD


def _byte_entropies(model, raw_bytes, device="cpu"):
    """raw_bytes: bytes. Returns a 1-D tensor of length len(raw_bytes), the
    predictive entropy (nats) for each real byte -- BOS/EOS positions are
    dropped from the returned tensor (see induce_boundaries)."""
    ids = [BOS_ID] + [b + OFFSET for b in raw_bytes] + [EOS_ID]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tokens).float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(-1)[0]  # (seq_len,) = BOS + n_bytes + EOS
    return entropies[1:-1]  # drop BOS/EOS positions, keep one entropy per real byte


def induce_boundaries(model, raw, device="cpu"):
    """model + one raw byte sequence (bytes/bytearray/str, or an already-built
    1-D LongTensor of raw byte VALUES, not token ids) -> 0/1 boundary-action
    list, same convention as every other systems/*/segment.py's
    induce_boundaries (action[i]==1 means byte i is the LAST byte of its
    patch; common.bytes_utils.spans_from_boundaries always closes a patch at
    the final position too, so the last byte needs no explicit 1).

    The entropy-model sequence includes BOS/EOS around the real bytes (see
    model.py's byte<->id mapping), which is why token indices 0/1 always
    start a patch in the paper's own rule -- BOS is its own token, and the
    first real byte (token index 1) has no real prior context either. Once
    BOS/EOS are dropped (_byte_entropies), that maps to: byte 0 always
    starts a patch (trivially true for every tokenizer's first byte), and
    byte i (i >= 1) starts a new patch iff entropies[i - 1] > threshold --
    entropies[i-1] here being _byte_entropies' i-1'th entry, i.e. the
    original token sequence's entropies[(i-1)+1] = entropies[i], which is
    exactly "the entropy of predicting byte i" as the module docstring's
    off-by-one note describes.
    """
    if torch.is_tensor(raw):
        raw_bytes = bytes(raw.detach().tolist())
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        raw_bytes = bytes(raw)

    n = len(raw_bytes)
    if n == 0:
        return []
    if n == 1:
        return [1]

    entropies = _byte_entropies(model, raw_bytes, device=device)  # length n
    starts = [False] * n
    starts[0] = True
    for i in range(1, n):
        if entropies[i - 1].item() > PATCHING_THRESHOLD:
            starts[i] = True

    actions = [0] * n
    for i in range(n - 1):
        if starts[i + 1]:
            actions[i] = 1  # byte i is the last byte of its patch
    return actions


def induce_spans(model, raw, device="cpu"):
    """model + one raw byte sequence -> list of byte-string patches, via
    common.bytes_utils.spans_from_boundaries (same span objects every other
    systems/*/segment.py's induce_spans produces, so this plugs directly into
    common.eval.cross_tokenizer.evaluate_on_groups)."""
    tensor = bytes_to_tensor(raw, device="cpu")  # CPU: byte VALUES only, not fed to the model directly here
    actions = induce_boundaries(model, raw, device=device)
    return spans_from_boundaries(tensor, actions)
