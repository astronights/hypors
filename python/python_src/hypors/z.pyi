from typing import Iterable

from .common import TailType, TestResult

def z_test(
    data: Iterable[float],
    pop_mean: float,
    pop_std: float,
    tail: TailType,
    alpha: float,
) -> TestResult:
    """One-sample z-test against a known population mean and standard deviation."""

def z_test_paired(
    data1: Iterable[float],
    data2: Iterable[float],
    pop_std_diff: float,
    tail: TailType,
    alpha: float,
) -> TestResult:
    """Paired z-test, given the population standard deviation of the differences."""

def z_test_ind(
    data1: Iterable[float],
    data2: Iterable[float],
    pop_std1: float,
    pop_std2: float,
    tail: TailType,
    alpha: float,
) -> TestResult:
    """Independent two-sample z-test with known population standard deviations."""

def z_sample_size(
    effect_size: float, alpha: float, power: float, std_dev: float, tail: TailType
) -> float:
    """Sample size required for a z-test at the given effect size, alpha and power."""
