"""Rényi efficiency validated against exact closed-form cases derived from the
formula itself (not copied from a source, since Zouhar et al.'s paper table
wasn't fully extractable):

  - a uniform distribution over v types has Rényi entropy exactly log(v) for
    ANY alpha, so efficiency is exactly 1.0 regardless of alpha.
  - a one-hot (fully degenerate) distribution has Rényi entropy exactly 0 for
    alpha != 1, so efficiency is exactly 0.0.
"""

from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency


def test_renyi_efficiency_uniform_is_one():
    freqs = [10] * 8
    assert abs(renyi_efficiency(freqs, alpha=2.5) - 1.0) < 1e-9


def test_renyi_efficiency_uniform_is_alpha_invariant():
    freqs = [5] * 16
    for alpha in (1.5, 2.0, 2.5, 3.0):
        assert abs(renyi_efficiency(freqs, alpha=alpha) - 1.0) < 1e-9


def test_renyi_efficiency_degenerate_is_zero():
    freqs = [100, 0, 0, 0]
    assert abs(renyi_efficiency(freqs, alpha=2.5)) < 1e-9


def test_renyi_efficiency_between_extremes():
    freqs = [50, 30, 15, 5]
    eff = renyi_efficiency(freqs, alpha=2.5)
    assert 0.0 < eff < 1.0


def test_gini_equal_values_is_zero():
    assert abs(gini_coefficient([4, 4, 4, 4])) < 1e-9


def test_gini_more_unequal_is_higher():
    low = gini_coefficient([5, 5, 5, 5.1])
    high = gini_coefficient([1, 1, 1, 100])
    assert high > low


def test_compression_rate_basic():
    assert compression_rate(num_bytes=10, num_tokens=5) == 2.0
    assert compression_rate(num_bytes=10, num_tokens=0) == 0.0
