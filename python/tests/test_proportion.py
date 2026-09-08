"""Fixtures ported from tests/proportion.rs."""

from conftest import EPSILON, SAMPLE_SIZE_TOL

from hypors import TailType
from hypors.proportion import prop_sample_size, z_test, z_test_ind

DATA1 = [1, 1, 1, 0, 0]
DATA2 = [1, 1, 0, 0, 0]


def test_z_test():
    result = z_test(DATA1, 0.5, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.447) < EPSILON
    assert abs(result.p_value - 0.655) < EPSILON
    assert result.null_hypothesis == "H0: p = 0.5"
    assert result.alt_hypothesis == "Ha: p ≠ 0.5"


def test_z_test_ind_unpooled():
    result = z_test_ind(DATA1, DATA2, TailType.Two, 0.05, False)
    assert abs(result.test_statistic - 0.645) < EPSILON
    assert abs(result.p_value - 0.518) < EPSILON
    assert result.null_hypothesis == "H0: p1 = p2"


def test_z_test_ind_pooled():
    result = z_test_ind(DATA1, DATA2, TailType.Two, 0.05, True)
    assert abs(result.test_statistic - 0.632) < EPSILON
    assert abs(result.p_value - 0.527) < EPSILON


def test_prop_sample_size():
    n = prop_sample_size(0.4, 0.6, 0.05, 0.80)
    assert abs(n - 97.0) < SAMPLE_SIZE_TOL
