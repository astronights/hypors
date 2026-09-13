"""Hypothesis testing for Python, backed by the hypors Rust crate.

Everything here is defined by the compiled extension; this module only
re-exports it so that the package can also carry type stubs (PEP 561).
"""

from .hypors import *  # noqa: F401,F403
from .hypors import __all__  # noqa: F401
