"""FANTA = "FAir MANTa": MantaModel's architecture, completely unchanged, trained
with an added cross-lingual fairness term in the loss -- see fanta/train.py.

MANTa itself has no reward/RL machinery and no rate-consistency loss at all (see
manta/model.py's own module docstring); its only loss is next-byte cross-entropy.
FANTA adds exactly one thing: a differentiable Gini-coefficient penalty over each
language's mean compression rate within a training batch, so cross-lingual
disparity is penalized directly during backprop -- structurally closer to
flexitokens/model.py's boundary_hinge_loss (a direct loss term, not RL reward-
shaping like fairtok's lambda_fair) but comparing languages to EACH OTHER (a Gini
penalty) rather than each language to a fixed target band.

Nothing about MantaModel itself changes for this: the only new differentiable
quantity FANTA needs (an expected-compression-rate proxy per sequence) already
exists in MantaOutput.mu, which manta/train.py's OWN diagnostic already reads --
just detached there, since it's only used for a console print. FANTA reads the
exact same tensor without detaching, so gradients flow from the fairness loss back
through the frontier predictor.
"""

from collections import defaultdict

import torch

from manta.model import MantaModel, next_byte_loss  # noqa: F401 -- re-exported so
# fanta.train can `from .model import MantaModel, next_byte_loss, ...` like every
# other tokenizer package does, without every caller needing to know FANTA and
# MANTa share one (unmodified) model implementation.


def differentiable_gini(values):
    """Differentiable torch port of common.metrics.gini_coefficient's exact closed
    form (sort + weighted sum) -- that numpy version is fine for periodic
    REPORTING (fairtok's fairness scalar, vocab-stat summaries), but numpy ops
    detach from autograd, so it can't be used inside a loss. `values`: a 1-D torch
    tensor, one differentiable scalar per language (see per_lang_compression_rate
    below). Gradient flows from the returned scalar back into every element of
    `values`.

    Degenerate cases (0 or 1 languages present, or a non-positive total -- the
    latter shouldn't happen in practice since compression rates are always
    positive, but is guarded anyway) return a zero that stays connected to the
    autograd graph (multiplying by 0.0 rather than returning a bare Python float),
    so summing this into a larger loss never breaks backprop even on a step where
    the batch happens to cover only one language.
    """
    n = values.numel()
    if n <= 1:
        return values.sum() * 0.0
    sorted_vals, _ = torch.sort(values)
    total = sorted_vals.sum()
    if total.item() <= 0:
        return values.sum() * 0.0
    index = torch.arange(1, n + 1, dtype=values.dtype, device=values.device)
    return (2 * (index * sorted_vals).sum() - (n + 1) * total) / (n * total)


def per_lang_compression_rate(langs, lengths, output, eps=1e-6):
    """langs: list[str], one per sequence in the batch (index-aligned with
    lengths). lengths: (B,) tensor of real (unpadded) byte counts. output:
    MantaOutput from this step's MantaModel(...) call -- NOT detached, unlike
    manta/train.py's own "mean_blocks/seq" diagnostic, since gradients need to
    flow from here back into the frontier predictor.

    Returns dict[lang -> 0-d tensor], one differentiable mean compression rate
    (bytes/expected-block, common.metrics.compression_rate's exact quantity) per
    language present in this batch, averaged over however many of that language's
    sequences this batch happened to include.

    Deliberately NOT common.metrics.compression_rate itself: that function's
    `if num_tokens <= 0` guard assumes a scalar, and raises on a multi-element
    tensor ("the truth value of a tensor with more than one element is
    ambiguous") -- the batched elementwise version here is the vectorized
    equivalent of the same formula.
    """
    B = lengths.shape[0]
    last_idx = (lengths - 1).clamp_min(0)
    expected_blocks = (
        output.mu.gather(1, last_idx.unsqueeze(1)).squeeze(1) + 1
    )  # (B,) -- NOT detached
    rates = lengths.to(expected_blocks.dtype) / (expected_blocks + eps)  # (B,)

    rates_by_lang = defaultdict(list)
    for i in range(B):
        rates_by_lang[langs[i]].append(rates[i])
    return {lang: torch.stack(vals).mean() for lang, vals in rates_by_lang.items()}


def fairness_loss(langs, lengths, output):
    """The actual FANTA loss term: differentiable_gini over
    per_lang_compression_rate's per-language means. Callers add
    `cfg.lambda_fair * fairness_loss(...)` to the plain next-byte CE loss (see
    fanta/train.py) -- mirrors flexitokens.model.boundary_hinge_loss's role
    (a direct, backprop-through loss term) but with no target-rate concept at
    all: this term is satisfied purely by languages compressing SIMILARLY to
    each other, not by any of them hitting a specific rate."""
    per_lang_rate = per_lang_compression_rate(langs, lengths, output)
    rates_tensor = torch.stack(list(per_lang_rate.values()))
    return differentiable_gini(rates_tensor), per_lang_rate
