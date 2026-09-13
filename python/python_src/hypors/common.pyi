from typing import Any, Dict, Tuple

class TailType:
    """Which tail of the distribution a test is run against."""

    Left: TailType
    """Test whether the statistic is smaller than expected."""
    Right: TailType
    """Test whether the statistic is larger than expected."""
    Two: TailType
    """Test for a difference in either direction."""

    @property
    def name(self) -> str:
        """The variant name, one of ``Left``, ``Right`` or ``Two``."""

class TestResult:
    """The outcome of a hypothesis test. Attributes are read-only."""

    def __init__(
        self,
        test_statistic: float,
        p_value: float,
        confidence_interval: Tuple[float, float],
        null_hypothesis: str,
        alt_hypothesis: str,
        reject_null: bool,
    ) -> None: ...
    @property
    def test_statistic(self) -> float:
        """The computed test statistic."""

    @property
    def p_value(self) -> float:
        """The p-value for the test."""

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Confidence interval, or ``(nan, nan)`` where the test defines none."""

    @property
    def null_hypothesis(self) -> str:
        """The null hypothesis being tested."""

    @property
    def alt_hypothesis(self) -> str:
        """The alternative hypothesis."""

    @property
    def reject_null(self) -> bool:
        """Whether the null hypothesis is rejected at the given alpha."""

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain dict, for serialisation or tabulating."""
