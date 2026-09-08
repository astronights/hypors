"""Fixtures ported from tests/anova.rs."""

from conftest import EPSILON, SAMPLE_SIZE_TOL

from hypors.anova import anova, f_sample_size


def test_anova():
    result = anova(
        [[2.0, 3.0, 3.0, 5.0, 6.0], [3.0, 4.0, 4.0, 6.0, 8.0], [5.0, 6.0, 7.0, 8.0, 9.0]],
        0.05,
    )
    assert abs(result.test_statistic - 4.261) < EPSILON
    assert abs(result.p_value - 0.039) < EPSILON
    assert result.reject_null
    assert result.null_hypothesis == "H0: µ1 = µ2 = µ3"
    assert result.alt_hypothesis == "Ha: At least one group mean is different"


def test_anova_no_rejection():
    result = anova(
        [[2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0, 8.0]],
        0.05,
    )
    assert abs(result.test_statistic - 2.0) < EPSILON
    assert abs(result.p_value - 0.177) < EPSILON
    assert not result.reject_null


def test_f_sample_size():
    n = f_sample_size(0.25, 0.05, 0.80, 3)
    assert abs(n - 21717.0) <= SAMPLE_SIZE_TOL
