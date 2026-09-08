"""Fixtures ported from tests/z.rs."""

from conftest import EPSILON, SAMPLE_SIZE_TOL

from hypors import TailType
from hypors.z import z_sample_size, z_test, z_test_ind, z_test_paired

DATA1 = [2.0, 3.0, 5.0, 7.0, 11.0]
DATA2 = [1.0, 3.0, 6.0, 7.0, 10.0]


def test_z_test():
    result = z_test(DATA1, 5.0, 2.0, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.670) < EPSILON
    assert abs(result.p_value - 0.502) < EPSILON
    assert result.null_hypothesis == "H0: µ = 5"
    assert abs(result.confidence_interval[0] - 3.846954) < EPSILON
    assert abs(result.confidence_interval[1] - 7.353045) < EPSILON


def test_z_test_paired():
    result = z_test_paired(DATA1, DATA2, 1.5, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.298) < EPSILON
    assert abs(result.p_value - 0.765) < EPSILON
    assert abs(result.confidence_interval[0] - -1.114783) < EPSILON
    assert abs(result.confidence_interval[1] - 1.514783) < EPSILON


def test_z_test_ind():
    result = z_test_ind(DATA1, DATA2, 2.0, 1.5, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.179) < EPSILON
    assert abs(result.p_value - 0.858) < EPSILON
    assert abs(result.confidence_interval[0] - -1.991306) < EPSILON
    assert abs(result.confidence_interval[1] - 2.391306) < EPSILON


def test_z_sample_size():
    n = z_sample_size(0.3, 0.05, 0.80, 1.0, TailType.Two)
    assert abs(n - 87.79) < SAMPLE_SIZE_TOL
