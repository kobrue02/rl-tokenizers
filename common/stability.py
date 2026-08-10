"""Boundary-stability diagnostic, shared across all four tokenizers in this repo.

Adapted from "Proxy Compression for Language Modeling" (Zheng et al. 2026)'s
"compressor stability" analysis (Sec 3.4): they measure how much a compressor's
output changes under a small random input perturbation (10% random deletion), and
find that instability -- large output changes from small input changes -- is exactly
why gzip-compressed training proxies fail to transfer while tokenizer/neural-
compressor proxies succeed. Repurposed here as a per-language fairness check: if a
trained boundary predictor (fairtok's RL policy, or MAGNET/FlexiTokens/MANTa's
Gumbel-sigmoid/soft-assignment predictors) is disproportionately unstable for some
languages, that's a real quality gap none of this project's other metrics
(compression rate, Rényi efficiency, Gini) would surface -- those score the
segmentation ACHIEVED, not how much it would change under a trivial edit.

Each tokenizer's own boundary/span-inducing entry point has a different signature
(fairtok.policy.segment_bytes needs a policy; magnet.segment.induce_spans needs an
extra `script` arg; flexitokens/manta's induce_spans take just (model, byte_seq)) --
this module stays agnostic to all of that by taking an already-bound
`bytes -> list[bytes] spans` callable per language, built by each tokenizer's own
cli.py.
"""

from collections import defaultdict

import numpy as np

from common.metrics import boundary_stability


def sequences_by_lang_from_groups(train_groups):
    """{lang: [text, text, ...]} pooled across every group -- the same raw text
    common.parity.compute_lang_parity_ratios reads, just regrouped by language
    instead of by group."""
    sequences_by_lang = defaultdict(list)
    for group in train_groups:
        for lang, text in group.items():
            sequences_by_lang[lang].append(text)
    return sequences_by_lang


def perturb_bytes(byte_seq, rng, delete_frac=0.1):
    """Delete a random `delete_frac` fraction of byte positions (at least one, if the
    sequence has more than one byte) -- the same style of perturbation the
    proxy-compression paper used. byte_seq: str/bytes/1-D LongTensor. Returns `bytes`."""
    if isinstance(byte_seq, str):
        raw = list(byte_seq.encode("utf-8"))
    elif hasattr(byte_seq, "detach"):
        raw = byte_seq.detach().cpu().tolist()
    else:
        raw = list(byte_seq)
    n = len(raw)
    if n <= 1:
        return bytes(raw)
    num_delete = min(max(1, int(round(n * delete_frac))), n - 1)  # keep >= 1 byte
    drop = set(rng.choice(n, size=num_delete, replace=False).tolist())
    return bytes(b for i, b in enumerate(raw) if i not in drop)


def sequence_stability(induce_spans_fn, text, rng, delete_frac=0.1):
    """induce_spans_fn: bytes -> list[bytes] spans, already bound by the caller to a
    specific model (+ script/language, if that tokenizer's own induce_spans needs
    one). text: str/bytes for ONE sentence. Returns a single stability score in [0, 1]
    (see common.metrics.boundary_stability) for that one sequence -- average over many
    sequences per language (see stability_by_lang) for a stable per-language reading."""
    original = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    spans_before = induce_spans_fn(original)
    perturbed = perturb_bytes(original, rng, delete_frac)
    if len(perturbed) == 0:
        return 1.0
    spans_after = induce_spans_fn(perturbed)
    return boundary_stability(spans_before, spans_after)


def stability_by_lang(
    induce_spans_fn_by_lang, sequences_by_lang, seed=0, delete_frac=0.1, num_samples=15
):
    """induce_spans_fn_by_lang: dict[lang -> (bytes -> list[bytes] spans)] callable,
    already bound to that language's model/script/etc. sequences_by_lang: dict[lang ->
    list of raw texts] to sample from (see sequences_by_lang_from_groups). Returns
    dict[lang -> mean stability score], averaged over up to num_samples randomly
    chosen sequences per language -- languages missing from either dict are skipped."""
    rng = np.random.default_rng(seed)
    result = {}
    for lang, texts in sequences_by_lang.items():
        induce_fn = induce_spans_fn_by_lang.get(lang)
        if induce_fn is None or not texts:
            continue
        idx = rng.choice(len(texts), size=min(num_samples, len(texts)), replace=False)
        scores = [
            sequence_stability(induce_fn, texts[i], rng, delete_frac) for i in idx
        ]
        result[lang] = float(np.mean(scores))
    return result
