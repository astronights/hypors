"""Shared tolerances, matching the Rust integration suite.

Every expected value in these tests is the value the corresponding Rust test
asserts, so the bindings are checked against the same fixtures rather than
against whatever they happen to return.
"""

EPSILON = 0.001  # tests/*.rs
MW_EPSILON = 0.0001  # tests/mann_whitney.rs
SAMPLE_SIZE_TOL = 1.0  # sample-size assertions in the Rust suite
