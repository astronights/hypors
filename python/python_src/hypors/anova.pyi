from typing import Iterable

from .common import TestResult

def anova(data_groups: Iterable[Iterable[float]], alpha: float) -> TestResult:
    """One-way ANOVA across two or more independent groups.

    ``data_groups`` is an iterable of groups, each itself an iterable of numbers.
    """

def f_sample_size(
    effect_size: float, alpha: float, power: float, num_groups: int
) -> float:
    """Sample size per group required for a one-way ANOVA."""
