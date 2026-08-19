"""Regression tests for systems.tokenization.flexitokens.train.derive_alpha_beta's
beta_floor -- confirmed live: on real BOUQuET training data, 258 of 259
languages' sentence-length coefficient of variation drove beta_L to EXACTLY 0,
permanently disabling boundary_hinge_loss's lower-bound term and letting a real
trained checkpoint collapse to ~0.6% boundary rate (avg 164 bytes/token). These
tests use synthetic high-variance groups to reproduce that exact collapse
condition without needing the real (gated, ~600-group) BOUQuET corpus."""

from systems.tokenization.flexitokens.train import derive_alpha_beta

# Deliberately skewed lengths (a few very long sentences among many short ones)
# -- mimics BOUQuET's real mix of short phrases and long paragraphs, which is
# what drove cv > 1/margin_lambda for ~99.6% of languages in practice.
_HIGH_VARIANCE_LENGTHS = ["a"] * 8 + ["a" * 200] * 2


def _make_groups(lengths_by_lang):
    """lengths_by_lang: {lang: [text, text, ...]} (all same length across
    langs) -> list of {lang: text} groups, one group per index."""
    n = len(next(iter(lengths_by_lang.values())))
    return [{lang: texts[i] for lang, texts in lengths_by_lang.items()} for i in range(n)]


def test_beta_never_collapses_to_zero_under_high_variance():
    groups = _make_groups({
        "eng": _HIGH_VARIANCE_LENGTHS,
        "xyz": _HIGH_VARIANCE_LENGTHS,
    })
    alpha_by_lang, beta_by_lang, _ = derive_alpha_beta(groups, anchor_lang="eng")

    assert beta_by_lang["eng"] >= 0.02
    assert beta_by_lang["xyz"] >= 0.02
    # the exact bug this regression-tests: before beta_floor existed, high cv
    # from this length distribution drove beta to precisely 0.0.
    assert beta_by_lang["eng"] > 0.0
    assert beta_by_lang["xyz"] > 0.0


def test_beta_never_exceeds_alpha():
    groups = _make_groups({"eng": _HIGH_VARIANCE_LENGTHS, "xyz": _HIGH_VARIANCE_LENGTHS})
    alpha_by_lang, beta_by_lang, _ = derive_alpha_beta(groups, anchor_lang="eng")
    for lang in alpha_by_lang:
        assert beta_by_lang[lang] <= alpha_by_lang[lang]


def test_beta_floor_is_configurable():
    groups = _make_groups({"eng": _HIGH_VARIANCE_LENGTHS, "xyz": _HIGH_VARIANCE_LENGTHS})
    _, beta_by_lang, _ = derive_alpha_beta(groups, anchor_lang="eng", beta_floor=0.1)
    assert all(b >= 0.1 for b in beta_by_lang.values())


def test_low_variance_language_keeps_a_meaningful_band_not_just_the_floor():
    """A language with LOW length variance should still get beta derived from
    the margin formula (not just clamped at the floor) -- beta_floor is a
    safety net for the high-variance collapse case, not a replacement for the
    paper's own formula when it behaves reasonably."""
    low_variance_lengths = ["a" * 10] * 10  # zero variance -> cv=0 -> beta == alpha
    groups = _make_groups({"eng": low_variance_lengths, "xyz": low_variance_lengths})
    alpha_by_lang, beta_by_lang, _ = derive_alpha_beta(groups, anchor_lang="eng")
    assert beta_by_lang["eng"] == alpha_by_lang["eng"]


def test_derive_alpha_beta_matches_confirmed_real_bouquet_collapse_shape():
    """Sanity check mirroring the exact diagnostic run against real BOUQuET
    data during investigation: with the fix, NO language should land at
    beta==0, whereas the pre-fix formula (max(0.0, ...) instead of
    max(beta_floor, ...)) put 258/259 real languages there."""
    groups = _make_groups({
        "eng": _HIGH_VARIANCE_LENGTHS,
        "chr": _HIGH_VARIANCE_LENGTHS,
        "arn": _HIGH_VARIANCE_LENGTHS,
        "mya": _HIGH_VARIANCE_LENGTHS,
    })
    _, beta_by_lang, _ = derive_alpha_beta(groups, anchor_lang="eng")
    assert all(b > 0.0 for b in beta_by_lang.values())
