"""Differentiable segment pooling ("dynamic token pooling" / Hourglass-style
downsample+upsample), ported directly from the MAGNET and FlexiTokens reference
implementations -- the two are byte-identical modulo whitespace/comments (both
papers build on the same "dynamic pooling" lineage: Nawrot et al., "Hierarchical
Transformers Are More Efficient Language Models" / "Dynamic Token Pooling"):
  https://github.com/orevaahia/magnet-tokenization/blob/main/src/shortening.py
  https://github.com/skai-research/flexitokens/blob/main/src/model/shortening.py
Reused directly with the project owner's explicit authorization (both repos are
public and intended for research reuse).

Adapted from the originals' TIME-MAJOR (T, B, D) hidden-state convention to this
project's BATCH-FIRST (B, T, D) convention (matches nn.TransformerEncoder's own
batch_first=True elsewhere in this repo) -- purely a tensor-layout change (einsum
equations relabeled to match), not a change to the algorithm itself. Verified
numerically equivalent to the original time-major functions on random inputs
before being wired into magnet/model.py and flexitokens/model.py (see git history
/ conversation for the verification script).

Why this replaces both models' previous scatter_add_/gather-based pooling: that
approach hardened `boundaries` into plain integer segment ids before pooling, and
integer indices carry no gradient -- so the reconstruction (next-byte) loss's
gradient could only reach the boundary predictor through a separate boundary-rate
loss term, never through the pooling operation itself. Here, `downsample`/
`upsample` build a fully differentiable (B, T, S) assignment matrix directly from
the continuous `boundaries` tensor (via a cumulative sum, never detached), so
gradient flows through the pooling itself back into the boundary predictor's
weights too -- a strictly more faithful reproduction of the reference mechanism.
"""

import torch


def _final(foo, upsample):
    """foo: (B, T, S) if not upsample, else (B, S, T) -- see `_common`. Turns the
    "which (position, segment) pairs belong together" zero/nonzero pattern into
    actual mean-pooling weights (each real pair gets 1/segment_size, zero
    elsewhere)."""
    autoregressive = foo != 0
    lel = 1 - foo
    lel[autoregressive] = 0
    dim = 2 if upsample else 1
    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)
    return lel


def _common(boundaries, upsample=False):
    """boundaries: (B, T) 0/1, boundaries[b, t]==1 means position t is the LAST
    byte of its segment. Returns None if no segment in the whole batch has any
    boundary at all (degenerate all-one-segment case), else a (B, T, S) tensor
    (S = the batch's max segment count, +1 more if upsample=True for the
    prepended null segment -- see `downsample`) whose zero entries mark which
    (position, segment) pairs belong together; _final turns this into weights."""
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if upsample:
        n_segments += 1
    if n_segments == 0:
        return None

    tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
        start=0, end=n_segments, device=boundaries.device
    )
    hh1 = boundaries.cumsum(1)
    if not upsample:
        hh1 = hh1 - boundaries  # a tensor counting 0..n_segments, reduced by 0 or 1
    return tmp - hh1.unsqueeze(-1)


def downsample(boundaries, hidden, null_group):
    """boundaries: (B, T) 0/1. hidden: (B, T, D). null_group: (1, 1, D) learned
    parameter, prepended as segment 0 -- `upsample`'s "segment i feeds byte
    positions in segment i+1, never its own" shift (see there) needs something to
    feed the FIRST real segment's own byte positions with, since there's no
    segment -1 to use instead.
    Returns (B, S+1, D) (+1 for the prepended null_group)."""
    foo = _common(boundaries, upsample=False)
    if foo is None:
        return null_group.repeat(hidden.size(0), 1, 1)
    bar = _final(foo, upsample=False)  # (B, T, S)
    shortened = torch.einsum("btd,bts->bsd", hidden, bar)
    return torch.cat([null_group.repeat(hidden.size(0), 1, 1), shortened], dim=1)


def upsample(boundaries, shortened_hidden):
    """boundaries: (B, T) 0/1 (the SAME tensor passed to downsample).
    shortened_hidden: (B, S+1, D) (downsample's output, including the prepended
    null_group). Segment i's pooled representation is broadcast only to segment
    (i+1)'s byte positions, never its own -- what stops a byte's upsampled
    representation from leaking information about its own segment's boundary
    decision (and therefore its own identity, for a next-byte-prediction loss)
    back into predicting itself.
    Returns (B, T, D)."""
    foo = _common(boundaries, upsample=True)
    bar = _final(foo, upsample=True)  # (B, T, S+1)
    return torch.einsum("bsd,bts->btd", shortened_hidden, bar)
