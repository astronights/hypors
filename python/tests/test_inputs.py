"""The bindings accept any iterable of numbers, and reject the rest clearly.

Nothing here is a dependency of the package: numpy and polars are only used
to show that third-party sequences work by virtue of being iterable.
"""

import pytest
from conftest import EPSILON

from hypors import TailType
from hypors.t import t_test

DATA = [2.0, 3.0, 5.0, 7.0, 11.0]
EXPECTED_STATISTIC = 0.374


def run(data):
    return t_test(data, 5.0, TailType.Two, 0.05).test_statistic


def test_list_and_tuple():
    assert abs(run(DATA) - EXPECTED_STATISTIC) < EPSILON
    assert abs(run(tuple(DATA)) - EXPECTED_STATISTIC) < EPSILON


def test_generator_and_range_and_ints():
    assert abs(run(x for x in DATA) - EXPECTED_STATISTIC) < EPSILON
    assert abs(run(iter(DATA)) - EXPECTED_STATISTIC) < EPSILON
    # Integers coerce to float, so an int sequence is accepted.
    assert run(range(1, 6)) == pytest.approx(t_test([1, 2, 3, 4, 5], 5.0, TailType.Two, 0.05).test_statistic)


def test_numpy_array():
    np = pytest.importorskip("numpy")
    assert abs(run(np.array(DATA)) - EXPECTED_STATISTIC) < EPSILON


def test_polars_series():
    pl = pytest.importorskip("polars")
    assert abs(run(pl.Series(DATA)) - EXPECTED_STATISTIC) < EPSILON


def test_non_iterable_is_rejected_by_name():
    with pytest.raises(ValueError, match="data must be an iterable of numbers"):
        t_test(5.0, 5.0, TailType.Two, 0.05)


def test_non_numeric_element_names_the_index():
    with pytest.raises(ValueError, match=r"data\[2\] must be a number"):
        t_test([1.0, 2.0, "three", 4.0], 5.0, TailType.Two, 0.05)


def test_nested_argument_names_the_row_and_index():
    from hypors.anova import anova

    with pytest.raises(ValueError, match=r"data_groups\[1\]\[0\] must be a number"):
        anova([[1.0, 2.0], ["a", "b"]], 0.05)


def test_expected_counts_reject_non_integers():
    from hypors.chi_square import chi2_sample_size_gof

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        chi2_sample_size_gof([20, -1, 50], 0.05)
