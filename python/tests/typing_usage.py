"""A strict-typed tour of the public API, checked by mypy in CI.

Not collected by pytest: its whole job is to fail type checking if the stubs
stop describing the package correctly. It doubles as a worked example of the
types each call expects and returns.
"""

from typing import Any, Dict, List, Tuple

from hypors import TailType, TestResult
from hypors.anova import anova, f_sample_size
from hypors.chi_square import (
    chi2_sample_size_gof,
    chi2_sample_size_ind,
    chi2_sample_size_variance,
    goodness_of_fit,
    independence,
    variance,
)
from hypors.mann_whitney import u_test
from hypors.proportion import prop_sample_size, z_test_ind
from hypors.t import t_sample_size, t_test, t_test_ind, t_test_paired
from hypors.z import z_sample_size, z_test, z_test_paired

SAMPLE: List[float] = [1.2, 2.3, 1.9, 2.5, 2.8]
OTHER: List[float] = [2.1, 2.9, 2.4, 3.1, 2.6]

# Any iterable of numbers is accepted, not only a list.
result: TestResult = t_test(SAMPLE, 2.0, TailType.Two, 0.05)
_: TestResult = t_test(tuple(SAMPLE), 2.0, TailType.Left, 0.05)
_ = t_test(range(5), 2.0, TailType.Right, 0.05)
_ = t_test((x for x in SAMPLE), 2.0, TailType.Two, 0.05)

statistic: float = result.test_statistic
p_value: float = result.p_value
interval: Tuple[float, float] = result.confidence_interval
null: str = result.null_hypothesis
alternative: str = result.alt_hypothesis
rejected: bool = result.reject_null
as_dict: Dict[str, Any] = result.to_dict()

tail_name: str = TailType.Two.name
rebuilt: TestResult = TestResult(1.0, 0.5, (0.1, 0.9), null, alternative, rejected)

_ = t_test_paired(SAMPLE, OTHER, TailType.Two, 0.05)
_ = t_test_ind(SAMPLE, OTHER, TailType.Two, 0.05, True)
_ = z_test(SAMPLE, 2.0, 0.5, TailType.Two, 0.05)
_ = z_test_paired(SAMPLE, OTHER, 0.5, TailType.Two, 0.05)
_ = z_test_ind(SAMPLE, OTHER, TailType.Two, 0.05, False)
_ = anova([SAMPLE, OTHER], 0.05)
_ = independence([[10.0, 20.0], [30.0, 40.0]], 0.05)
_ = goodness_of_fit([10.0, 20.0], [15.0, 15.0], 0.05)
_ = variance(SAMPLE, 1.0, TailType.Right, 0.05)
_ = u_test(SAMPLE, OTHER, 0.05, TailType.Two)

sizes: List[float] = [
    t_sample_size(0.5, 0.05, 0.8, 1.0, TailType.Two),
    z_sample_size(0.5, 0.05, 0.8, 1.0, TailType.Two),
    prop_sample_size(0.4, 0.5, 0.05, 0.8),
    f_sample_size(0.25, 0.05, 0.8, 3),
    chi2_sample_size_gof([10, 20, 30], 0.05),
    chi2_sample_size_ind([10, 20, 30], 0.05),
    chi2_sample_size_variance(0.5, 0.05, 0.8, 1.0),
]
