"""Fixtures ported from tests/t.rs."""

from conftest import EPSILON, SAMPLE_SIZE_TOL

from hypors import TailType
from hypors.t import t_sample_size, t_test, t_test_ind, t_test_paired

DATA1 = [2.0, 3.0, 5.0, 7.0, 11.0]
DATA2 = [1.0, 3.0, 6.0, 7.0, 10.0]


def test_t_test():
    result = t_test(DATA1, 5.0, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.374) < EPSILON
    assert abs(result.p_value - 0.726) < EPSILON
    assert not result.reject_null
    assert result.null_hypothesis == "H0: µ = 5"
    assert result.alt_hypothesis == "Ha: µ ≠ 5"
    assert abs(result.confidence_interval[0] - 1.157687) < EPSILON
    assert abs(result.confidence_interval[1] - 10.042312) < EPSILON


def test_t_test_paired():
    result = t_test_paired(DATA1, DATA2, TailType.Two, 0.05)
    assert abs(result.test_statistic - 0.534) < EPSILON
    assert abs(result.p_value - 0.621) < EPSILON
    assert result.null_hypothesis == "H0: µ1 = µ2"
    assert abs(result.confidence_interval[0] - -0.838850) < EPSILON
    assert abs(result.confidence_interval[1] - 1.238850) < EPSILON


def test_t_test_ind_unpooled():
    result = t_test_ind(DATA1, DATA2, TailType.Two, 0.05, False)
    assert abs(result.test_statistic - 0.089) < EPSILON
    assert abs(result.p_value - 0.931) < EPSILON
    assert abs(result.confidence_interval[0] - -4.967041) < EPSILON
    assert abs(result.confidence_interval[1] - 5.367041) < EPSILON


def test_t_test_ind_pooled():
    result = t_test_ind(DATA1, DATA2, TailType.Two, 0.05, True)
    assert abs(result.test_statistic - 0.089) < EPSILON
    assert abs(result.p_value - 0.931) < EPSILON
    # The pooled interval differs from the unpooled one in the fourth decimal.
    assert abs(result.confidence_interval[0] - -4.966684) < EPSILON
    assert abs(result.confidence_interval[1] - 5.366684) < EPSILON


def test_t_sample_size():
    n = t_sample_size(0.3, 0.05, 0.80, 1.0, TailType.Two)
    assert abs(n - 88.49) < SAMPLE_SIZE_TOL
