"""FANTA = "FAir MANTa": MantaModel's architecture, unchanged, trained with two
added loss terms (see fanta/train.py):

  1. A differentiable Gini-coefficient penalty over each language's mean
     compression rate within a batch (differentiable_gini,
     per_lang_compression_rate, fairness_loss below) -- penalizes cross-lingual
     disparity directly during backprop.
  2. A per-language rate ANCHOR (rate_anchor_loss) pulling each language's rate
     toward its own target (derived like fairtok's target_rate_by_lang, via
     common.eval.parity.compute_lang_parity_ratios), rather than one global
     target. Needed because term 1 alone has a degenerate solution: all
     languages compressing equally badly (collapsed to ~1 byte/token) also
     has Gini~=0 -- confirmed empirically (an early anchor-less run collapsed
     mean_compression_rate to ~1.0 within 10 steps). The anchor makes "fair"
     mean "similarly well-compressed," not "similarly uncompressed."

Both are direct differentiable loss terms (not RL reward-shaping like
fairtok's lambda_fair/target_rate) -- structurally closer to
flexitokens/model.py's boundary_hinge_loss.

MantaModel itself is unchanged: the expected-compression-rate proxy FANTA
needs is MantaOutput.mu, the same tensor manta/train.py's diagnostic reads
(but detached there, for a console print only). FANTA reads it without
detaching so gradients flow back through the frontier predictor.
"""

from collections import defaultdict

import torch

from systems.manta.model import MantaModel, next_byte_loss  # noqa: F401 -- re-exported
# so fanta.train can import from .model like every other tokenizer package.


def differentiable_gini(values):
    """Differentiable torch port of common.eval.metrics.gini_coefficient's
    closed form (sort + weighted sum) -- the numpy version detaches from
    autograd, so it can't be used inside a loss. `values`: 1-D tensor, one
    scalar per language.

    Degenerate cases (<=1 language, or non-positive total) return a zero that
    stays connected to the autograd graph (`values.sum() * 0.0`, not a bare
    float), so summing this into a larger loss never breaks backprop.
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
    """langs: list[str], index-aligned with lengths (B,). output: MantaOutput
    from this step's forward pass -- NOT detached, since gradients need to
    reach the frontier predictor.

    Returns dict[lang -> 0-d tensor], one differentiable mean compression rate
    (bytes/expected-block, same quantity as common.eval.metrics.compression_rate)
    per language present in the batch.

    Not common.eval.metrics.compression_rate itself: its `if num_tokens <= 0`
    guard assumes a scalar and raises on a multi-element tensor. This is the
    vectorized equivalent of the same formula.
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
    `cfg.lambda_fair * fairness_loss(...)` to the total loss (fanta/train.py).
    Satisfied by languages compressing similarly to each other, not by any
    hitting a specific rate -- rate_anchor_loss below adds that (needed to
    rule out the degenerate "similarly uncompressed" solution, see module
    docstring)."""
    per_lang_rate = per_lang_compression_rate(langs, lengths, output)
    rates_tensor = torch.stack(list(per_lang_rate.values()))
    return differentiable_gini(rates_tensor), per_lang_rate


def rate_anchor_loss(per_lang_rate, target_rate_by_lang, eps=1e-6):
    """The anchor half of FANTA's loss: pulls each language's mean compression
    rate this batch (per_lang_rate, not recomputed here) toward its own
    target (target_rate_by_lang, derived once via
    common.eval.parity.compute_lang_parity_ratios -- same mechanism
    fairtok.train uses, motivated by "Compute Optimal Tokenization"
    (Limisiewicz et al. 2026): compute-optimal compression rate is
    language-dependent, correlating with byte-length parity vs. an anchor
    language, not one global rate).

    Penalty is a squared log-ratio, `(log(rate) - log(target))**2`, not a
    plain squared difference: target rates differ many-fold across languages
    (~3 bytes/token for English vs. ~8+ for verbose scripts), so a plain
    squared error would weight high-target languages' errors more heavily for
    the same relative deviation. Log-ratio treats "20% off target" the same
    regardless of scale.

    Languages in per_lang_rate but missing from target_rate_by_lang are
    skipped (defensive; shouldn't happen since both derive from the same
    train_groups). Returns an autograd-connected zero if no language has a
    target, same convention as differentiable_gini.
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
