from typing import Iterable

from .common import TailType, TestResult

def u_test(
    data1: Iterable[float],
    data2: Iterable[float],
    alpha: float,
    tail_type: TailType,
) -> TestResult:
    """Mann-Whitney U test for two independent samples.

    The p-value matches ``scipy.stats.mannwhitneyu`` with ``method="asymptotic"``.
    ``test_statistic`` is ``min(U1, U2)`` while the p-value derives from ``U1``, so
    compare p-values rather than statistics when checking against scipy.
    """
