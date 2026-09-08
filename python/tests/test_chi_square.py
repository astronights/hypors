"""Fixtures ported from tests/chi_square.rs."""

import pytest
from conftest import EPSILON, SAMPLE_SIZE_TOL

from hypors import TailType
from hypors.chi_square import (
    chi2_sample_size_gof,
    chi2_sample_size_ind,
    chi2_sample_size_variance,
    goodness_of_fit,
    independence,
    variance,
)


def test_variance():
    result = variance([2.0, 3.0, 5.0, 7.0, 11.0], 5.0, TailType.Two, 0.05)
    assert abs(result.test_statistic - 10.24) < EPSILON
    assert abs(result.p_value - 0.073) < EPSILON
    assert result.null_hypothesis == "H0: σ² = 5"
    assert not result.reject_null


def test_variance_errors():
    # Mirrors the Rust suite: no data and too little data are distinct.
    with pytest.raises(ValueError, match="empty"):
        variance([], 5.0, TailType.Two, 0.05)
    with pytest.raises(ValueError, match="[Ii]nsufficient"):
        variance([1.0], 5.0, TailType.Two, 0.05)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="positive finite"):
            variance([1.0, 2.0, 3.0], bad, TailType.Two, 0.05)


def test_independence():
    result = independence([[20.0, 30.0], [50.0, 10.0]], 0.05)
    assert abs(result.test_statistic - 22.131) < EPSILON
    assert abs(result.p_value - 0.000) < EPSILON
    assert result.null_hypothesis == "H0: Variables are independent"
    assert result.reject_null


def test_goodness_of_fit():
    result = goodness_of_fit([30.0, 10.0, 20.0], [25.0, 15.0, 20.0], 0.05)
    assert abs(result.test_statistic - 2.666) < EPSILON
    assert abs(result.p_value - 0.263) < EPSILON
    assert not result.reject_null


def test_chi2_sample_sizes():
    assert abs(chi2_sample_size_gof([20, 30, 50], 0.05) - 480.0) < SAMPLE_SIZE_TOL
    assert abs(chi2_sample_size_ind([20, 30, 50], 0.05) - 480.0) < SAMPLE_SIZE_TOL
    assert abs(chi2_sample_size_variance(0.5, 0.05, 0.80, 1.0) - 121.0) < SAMPLE_SIZE_TOL
