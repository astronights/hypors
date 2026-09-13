from typing import Iterable

from .common import TailType, TestResult

def independence(
    contingency_table: Iterable[Iterable[float]], alpha: float
) -> TestResult:
    """Chi-square test of independence over a contingency table.

    ``contingency_table`` is an iterable of rows, each an iterable of counts.
    """

def goodness_of_fit(
    observed: Iterable[float], expected: Iterable[float], alpha: float
) -> TestResult:
    """Chi-square goodness-of-fit test comparing observed against expected counts."""

def variance(
    data: Iterable[float], pop_variance: float, tail: TailType, alpha: float
) -> TestResult:
    """Chi-square test for a sample variance against a known population variance."""

def chi2_sample_size_gof(expected_counts: Iterable[int], alpha: float) -> float:
    """Sample size for a goodness-of-fit test."""

def chi2_sample_size_ind(expected_counts: Iterable[int], alpha: float) -> float:
    """Sample size for a test of independence."""

def chi2_sample_size_variance(
    effect_size: float, alpha: float, power: float, variance: float
) -> float:
    """Sample size for a variance test."""
