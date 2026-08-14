"""FANTA = "FAir MANTa": MantaModel's architecture, completely unchanged, trained
with two added terms in the loss -- see fanta/train.py.

MANTa itself has no reward/RL machinery and no rate-consistency loss at all (see
manta/model.py's own module docstring); its only loss is next-byte cross-entropy.
FANTA adds:

  1. A differentiable Gini-coefficient penalty over each language's mean
     compression rate within a training batch (differentiable_gini,
     per_lang_compression_rate, fairness_loss below), so cross-lingual disparity
     is penalized directly during backprop.
  2. A per-language rate ANCHOR (rate_anchor_loss below) -- pulls each language's
     rate toward its own target, derived the same way fairtok's
     target_rate_by_lang is (common.eval.parity.compute_lang_parity_ratios), rather
     than a single global target. This exists because term 1 alone has a
     degenerate solution: every language compressing equally BADLY (all
     collapsed toward ~1 byte/token) also has Gini~=0. Confirmed empirically,
     not just theoretically -- an early FANTA training run (no rate anchor)
     collapsed mean_compression_rate to ~1.0 within 10 steps, satisfying the
     Gini term while producing no real compression or fair TREATMENT of
     anything. The anchor is what makes "fair" mean "similarly well-compressed,"
     not "similarly uncompressed."

Both terms are direct, differentiable loss terms (not RL reward-shaping like
fairtok's own lambda_fair/target_rate) -- structurally closer to
flexitokens/model.py's boundary_hinge_loss, except comparing languages to EACH
OTHER (the Gini term) in addition to each language having its own target (the
anchor term, same as flexitokens' alpha_L/beta_L bands).

Nothing about MantaModel itself changes for any of this: the only new
differentiable quantity FANTA needs (an expected-compression-rate proxy per
sequence) already exists in MantaOutput.mu, which manta/train.py's OWN diagnostic
already reads -- just detached there, since it's only used for a console print.
FANTA reads the exact same tensor without detaching, so gradients flow from both
loss terms back through the frontier predictor.
"""

from collections import defaultdict

import torch

from systems.manta.model import MantaModel, next_byte_loss  # noqa: F401 -- re-exported so
# fanta.train can `from .model import MantaModel, next_byte_loss, ...` like every
# other tokenizer package does, without every caller needing to know FANTA and
# MANTa share one (unmodified) model implementation.


def differentiable_gini(values):
    """Differentiable torch port of common.eval.metrics.gini_coefficient's exact closed
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
    (bytes/expected-block, common.eval.metrics.compression_rate's exact quantity) per
    language present in this batch, averaged over however many of that language's
    sequences this batch happened to include.

    Deliberately NOT common.eval.metrics.compression_rate itself: that function's
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
    """The Gini half of FANTA's loss: differentiable_gini over
    per_lang_compression_rate's per-language means. Callers add
    `cfg.lambda_fair * fairness_loss(...)` to the total loss (see
    fanta/train.py) -- satisfied purely by languages compressing SIMILARLY to
    each other, not by any of them hitting a specific rate (see
    rate_anchor_loss below for the term that adds the latter, needed to rule
    out the degenerate "similarly uncompressed" solution -- see module
    docstring)."""
    per_lang_rate = per_lang_compression_rate(langs, lengths, output)
    rates_tensor = torch.stack(list(per_lang_rate.values()))
    return differentiable_gini(rates_tensor), per_lang_rate


def rate_anchor_loss(per_lang_rate, target_rate_by_lang, eps=1e-6):
    """The anchor half of FANTA's loss: pulls each language's mean compression
    rate this batch (per_lang_rate, from fairness_loss/per_lang_compression_rate
    above -- NOT recomputed here) toward that language's own target
    (target_rate_by_lang, derived once at the start of training by
    fanta.train.FantaTrainer.train via common.eval.parity.compute_lang_parity_ratios
    -- the exact same per-language-target mechanism fairtok.train's
    target_rate_by_lang uses, motivated by the same finding: "Compute Optimal
    Tokenization" (Limisiewicz et al. 2026) shows the compute-optimal
    compression rate is language-dependent, correlating with each language's
    byte-length parity vs. an anchor language, not one global rate).

    Penalty is a SQUARED LOG-RATIO, `(log(rate) - log(target))**2`, not a plain
    squared difference: target rates can differ by many-fold across languages
    (e.g. ~3 bytes/token for English vs. ~8+ for a verbose script, per the real
    per-language spread fairtok's own target_rate_by_lang produces on this
    project's 9-language panel) -- a plain squared error would weight
    high-target-rate languages' errors far more heavily for the SAME relative
    deviation. The log-ratio form treats "20% off target" the same regardless
    of the target's absolute scale.

    Languages present in per_lang_rate but missing from target_rate_by_lang
    (shouldn't happen in practice, since both are derived from the same
    train_groups, but guarded defensively) are skipped rather than erroring.
    Returns a scalar that stays connected to the autograd graph even if no
    language in this batch has a target (an even-more-defensive edge case),
    the same convention differentiable_gini uses for its own degenerate cases.
    """
    sq_log_diffs = []
    for lang, rate in per_lang_rate.items():
        target = target_rate_by_lang.get(lang)
        if target is None:
            continue
        target_t = torch.tensor(target, dtype=rate.dtype, device=rate.device)
        log_diff = torch.log(rate.clamp_min(eps)) - torch.log(target_t.clamp_min(eps))
        sq_log_diffs.append(log_diff**2)
    if not sq_log_diffs:
        return next(iter(per_lang_rate.values())).sum() * 0.0
    return torch.stack(sq_log_diffs).mean()
