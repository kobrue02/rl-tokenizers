"""Morphological Edit Distance (MED) and Morphological Consistency F1, per Asgari
et al.'s "MorphBPE" (arXiv:2502.00894), Sec 3.3 (ii)/(iii) -- how closely a
tokenizer's own segmentation tracks true morpheme boundaries, and how consistently
it segments the same morpheme across different word contexts.

Structurally separate from common.eval.cross_tokenizer.evaluate_on_groups rather
than a new parameter on it: that function is hard-wired to document-shaped
eval_groups (list[dict[lang -> text]]) accumulating corpus-level stats, whereas
these metrics need per-WORD gold morpheme boundaries -- a completely different
input shape, only available for a SUBSET of languages (see
common.data.prepare_morphology_gold's own docstring for exactly which, and why).

BOTH metrics are prose-only in the cited paper -- no equations, no pseudocode (a
"pairwise alignment score based on dynamic programming" for MED; a word-pair
relational check with an underspecified k-means+bootstrap sampling procedure for
Consistency F1). The formalizations below are this project's own concrete,
documented choices where the paper is silent, not verified-identical
reproductions of undisclosed paper internals -- see each function's own
docstring for exactly what's a paper quote vs. a filled-in gap.

For English/Russian/Hungarian specifically, common.data.prepare_morphology_gold
sources gold data from the SIGMORPHON 2022 segmentation shared task -- the SAME
gold source the MorphBPE paper itself uses for those three languages (confirmed
via direct reading of that paper) -- so THOSE three languages' numbers are
directly comparable to MorphBPE's own reported Table 2 results, not just
same-metric-name coincidentally-different-data.
"""

from collections import defaultdict

import random

from common.eval.metrics import _atomic_sequence_edit_distance


def morphological_edit_distance(gold_morphs, predicted_spans):
    """Raw (UNNORMALIZED) edit distance between a word's gold morpheme sequence
    and a tokenizer's own predicted span sequence, both as lists of UTF-8 byte
    strings compared as atomic units (see _atomic_sequence_edit_distance) --
    exact substring equality per position, not a character/byte-offset
    alignment, so gold_morphs and predicted_spans just need to be encodable to
    bytes consistently; no shared offset space is required.

    Unnormalized deliberately, matching the paper's own explicit choice ("while
    it can be normalized by the number of morphemes in each word, we retain its
    raw form to provide a clearer indication of the average number of edits
    required") -- callers wanting a length-normalized version should divide by
    len(gold_morphs) themselves; this function doesn't bake in an assumption of
    which normalization (if any) a given comparison wants.
    """
    return _atomic_sequence_edit_distance(gold_morphs, predicted_spans)


def morphological_consistency_f1(gold_and_predicted_by_word, sample_pairs=2000, n_bootstrap=1000, seed=0):
    """gold_and_predicted_by_word: {word: (gold_morphs, predicted_spans)} for ONE
    language, both as lists of byte-string atomic units (same convention as
    morphological_edit_distance).

    Per the paper's own words (Sec 3.3(iii)): "[the measure] ensures that words
    sharing the same morphemes also share tokens (recall score) and that words
    with shared tokens correspondingly share morphemes (precision score) ...
    checking whether shared morphemes correspond to shared tokens and vice
    versa ... we use their harmonic mean". Concretely (this project's own fully-
    specified reading of that prose, since the paper gives no formula): for a
    sampled pair of distinct words (w1, w2),
      - a "morpheme-shared" pair is one where gold_morphs(w1) and gold_morphs(w2)
        have at least one morph in common;
      - a "token-shared" pair is one where predicted_spans(w1) and
        predicted_spans(w2) have at least one span in common;
      - recall = P(token-shared | morpheme-shared) over sampled pairs;
      - precision = P(morpheme-shared | token-shared) over sampled pairs;
      - F1 = harmonic mean of precision and recall (0.0 if both are 0).

    sample_pairs: the paper's own k-means(k=100)+bootstrap(N=10) sampling
    procedure never specifies its clustering feature space, so isn't
    reproducible as written -- this instead samples up to `sample_pairs` word
    PAIRS uniformly at random (or every pair, if fewer exist than the cap),
    a simpler, fully-specified alternative.

    F1's own confidence interval is computed by resampling PAIRS (not calling
    common.eval.metrics.bootstrap_ci on precision/recall independently, which
    would treat them as unrelated when they're computed from overlapping pairs
    of the same underlying sample) -- each of n_bootstrap resamples draws
    len(pairs) pairs with replacement and recomputes precision/recall/F1 from
    that resample, giving a genuine joint bootstrap over the compound statistic.

    Returns {"precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "n_pairs"}.
    All zeros (n_pairs=0) if fewer than 2 words are available to pair.
    """
    words = list(gold_and_predicted_by_word)
    n_words = len(words)
    if n_words < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f1_ci_low": 0.0, "f1_ci_high": 0.0, "n_pairs": 0}

    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(n_words) for j in range(i + 1, n_words)]
    pairs = rng.sample(all_pairs, sample_pairs) if len(all_pairs) > sample_pairs else all_pairs

    # Precompute (morpheme_shared, token_shared) once per pair -- reused both
    # for the point estimate and every bootstrap resample below.
    shared_flags = []
    for i, j in pairs:
        gold_i, pred_i = gold_and_predicted_by_word[words[i]]
        gold_j, pred_j = gold_and_predicted_by_word[words[j]]
        shared_flags.append((bool(set(gold_i) & set(gold_j)), bool(set(pred_i) & set(pred_j))))

    def _precision_recall_f1(flags):
        morpheme_shared_n = sum(1 for m, _ in flags if m)
        token_shared_n = sum(1 for _, t in flags if t)
        recall_hits = sum(1 for m, t in flags if m and t)
        precision_hits = sum(1 for m, t in flags if t and m)
        recall = recall_hits / morpheme_shared_n if morpheme_shared_n else 0.0
        precision = precision_hits / token_shared_n if token_shared_n else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    precision, recall, f1 = _precision_recall_f1(shared_flags)

    n = len(shared_flags)
    resampled_f1s = sorted(
        _precision_recall_f1(rng.choices(shared_flags, k=n))[2] for _ in range(n_bootstrap)
    )
    alpha = (1 - 0.95) / 2
    low_idx = int(alpha * n_bootstrap)
    high_idx = min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap))

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "f1_ci_low": resampled_f1s[low_idx], "f1_ci_high": resampled_f1s[high_idx],
        "n_pairs": len(pairs),
    }


def evaluate_morphological_segmentation(induce_spans_fn_by_lang, gold_words_by_lang):
    """Top-level entry point. induce_spans_fn_by_lang: dict[lang -> (bytes ->
    list[bytes] spans)] callable (the SAME per-language callables
    common.eval.cross_tokenizer.evaluate_on_groups takes -- confirmed these work
    unmodified on a single word's bytes, not just full documents, since none of
    bpe/manta/hf_frontier's own induce_spans wrappers assume document shape).

    gold_words_by_lang: dict[lang -> dict[word -> (gold_morphs, source)]], the
    shape common.data.prepare_morphology_gold's own loader produces -- gold_morphs
    is list[str] (raw morph substrings; encoded to bytes here, not by the
    caller), source is "morphynet"/"unisegments"/"sigmorphon2022segmentation"/
    "unimorph_aligned_approx" (see that module's own docstring for what each
    means and which is a true-gold vs. heuristic-approximation source).

    Skips (not errors on) a language present in gold_words_by_lang but missing
    from induce_spans_fn_by_lang -- same "checkpoints trained on different
    language sets is expected, not exceptional" convention evaluate_on_groups
    already uses.

    Returns {lang: {"med": float (mean over words), "consistency_f1": float,
    "consistency_f1_ci_low": float, "consistency_f1_ci_high": float,
    "precision": float, "recall": float, "n_words": int, "source": str}}.
    "source" is whichever source label the language's gold words carry (all
    words for one language always share one source, by construction of
    common.data.prepare_morphology_gold's own per-language output files).
    """
    results = {}
    for lang, gold_words in gold_words_by_lang.items():
        induce_fn = induce_spans_fn_by_lang.get(lang)
        if induce_fn is None or not gold_words:
            continue

        med_total = 0.0
        n_words = 0
        source = None
        gold_and_predicted_by_word = {}
        for word, (gold_morphs, word_source) in gold_words.items():
            source = word_source
            gold_bytes = [m.encode("utf-8") for m in gold_morphs]
            predicted_spans = induce_fn(word.encode("utf-8"))
            med_total += morphological_edit_distance(gold_bytes, predicted_spans)
            n_words += 1
            gold_and_predicted_by_word[word] = (gold_bytes, predicted_spans)

        consistency = morphological_consistency_f1(gold_and_predicted_by_word)
        results[lang] = {
            "med": med_total / n_words if n_words else 0.0,
            "consistency_f1": consistency["f1"],
            "consistency_f1_ci_low": consistency["f1_ci_low"],
            "consistency_f1_ci_high": consistency["f1_ci_high"],
            "precision": consistency["precision"],
            "recall": consistency["recall"],
            "n_words": n_words,
            "source": source,
        }
    return results
