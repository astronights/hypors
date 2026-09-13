from typing import Iterable

from .common import TailType, TestResult

def z_test(
    data: Iterable[float], pop_proportion: float, tail: TailType, alpha: float
) -> TestResult:
    """One-sample proportion z-test against a known population proportion."""

def z_test_ind(
    data1: Iterable[float],
    data2: Iterable[float],
    tail: TailType,
    alpha: float,
    pooled: bool,
) -> TestResult:
    """Two-sample proportion z-test; ``pooled`` selects the pooled-variance form."""

def prop_sample_size(p1: float, p2: float, alpha: float, power: float) -> float:
    """Sample size required to detect a difference between two proportions."""
