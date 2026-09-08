"""Fixtures ported from tests/mann_whitney.rs.

These values were regenerated against scipy 1.18.0; the bindings must return
them unchanged, since the whole point is that Python and Rust agree.
"""

import math

import pytest
from conftest import MW_EPSILON

from hypors import TailType
from hypors.mann_whitney import u_test


def test_u_test():
    result = u_test([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0, 7.0], 0.05, TailType.Two)
    assert abs(result.test_statistic - 4.5) < MW_EPSILON
    assert abs(result.p_value - 0.1138463) < MW_EPSILON
    assert not result.reject_null
    assert result.null_hypothesis == "H0: The distributions of both groups are equal."


def test_u_test_equal():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = u_test(data, data, 0.05, TailType.Two)
    assert abs(result.test_statistic - 12.5) < MW_EPSILON
    assert abs(result.p_value - 1.0) < MW_EPSILON


def test_u_test_one_sided_direction():
    larger = [5.0, 6.0, 7.0, 8.0, 9.0]
    smaller = [1.0, 2.0, 3.0, 4.0, 5.0]

    right = u_test(larger, smaller, 0.05, TailType.Right)
    assert abs(right.p_value - 0.0079853) < MW_EPSILON
    assert right.reject_null

    left = u_test(larger, smaller, 0.05, TailType.Left)
    assert abs(left.p_value - 0.9955921) < MW_EPSILON
    assert not left.reject_null


def test_u_test_one_sided_overlapping():
    data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    data2 = [3.0, 4.0, 5.0, 6.0, 7.0]
    assert abs(u_test(data1, data2, 0.05, TailType.Right).p_value - 0.9634301) < MW_EPSILON
    assert abs(u_test(data1, data2, 0.05, TailType.Left).p_value - 0.0569231) < MW_EPSILON


def test_u_test_heavy_ties():
    result = u_test(
        [1.0, 2.0, 2.0, 3.0, 3.0, 3.0], [2.0, 3.0, 3.0, 4.0, 4.0, 5.0], 0.05, TailType.Two
    )
    assert abs(result.test_statistic - 7.0) < MW_EPSILON
    assert abs(result.p_value - 0.0784029) < MW_EPSILON
    assert not result.reject_null


def test_u_test_all_tied():
    tied = [2.0, 2.0, 2.0]
    two = u_test(tied, tied, 0.05, TailType.Two)
    assert math.isnan(two.p_value)
    assert not two.reject_null

    for tail in (TailType.Right, TailType.Left):
        result = u_test(tied, tied, 0.05, tail)
        assert abs(result.p_value - 1.0) < MW_EPSILON
        assert not result.reject_null


def test_u_test_empty_group():
    with pytest.raises(ValueError, match="empty"):
        u_test([], [1.0, 2.0], 0.05, TailType.Two)
    with pytest.raises(ValueError, match="empty"):
        u_test([1.0, 2.0], [], 0.05, TailType.Two)
