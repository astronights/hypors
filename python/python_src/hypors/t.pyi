from typing import Iterable

from .common import TailType, TestResult

def t_test(
    data: Iterable[float], pop_mean: float, tail: TailType, alpha: float
) -> TestResult:
    """One-sample t-test against a known population mean."""

def t_test_paired(
    data1: Iterable[float], data2: Iterable[float], tail: TailType, alpha: float
) -> TestResult:
    """Paired two-sample t-test."""

def t_test_ind(
    data1: Iterable[float],
    data2: Iterable[float],
    tail: TailType,
    alpha: float,
    pooled: bool,
) -> TestResult:
    """Independent two-sample t-test; ``pooled`` selects the equal-variance form."""

def t_sample_size(
    effect_size: float, alpha: float, power: float, std_dev: float, tail: TailType
) -> float:
    """Sample size required for a t-test at the given effect size, alpha and power."""
