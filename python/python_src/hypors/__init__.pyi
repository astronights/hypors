from . import anova as anova
from . import chi_square as chi_square
from . import common as common
from . import mann_whitney as mann_whitney
from . import proportion as proportion
from . import t as t
from . import z as z
from .common import TailType as TailType
from .common import TestResult as TestResult

__version__: str

__all__ = [
    "__version__",
    "common",
    "anova",
    "chi_square",
    "mann_whitney",
    "proportion",
    "t",
    "z",
    "TailType",
    "TestResult",
]
