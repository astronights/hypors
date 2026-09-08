//! The Mann-Whitney U test.

use crate::convert::{to_f64_vec, to_py_err};
use crate::types::{TailType, TestResult};
use pyo3::prelude::*;

/// Mann-Whitney U test for two independent samples.
///
/// The p-value matches `scipy.stats.mannwhitneyu` with `method="asymptotic"`.
/// `test_statistic` is `min(U1, U2)` while the p-value derives from `U1`, so
/// compare p-values rather than statistics when checking against scipy.
#[pyfunction]
#[pyo3(signature = (data1, data2, alpha, tail_type))]
fn u_test(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    alpha: f64,
    tail_type: TailType,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::mann_whitney::u_test(data1, data2, alpha, tail_type.into())
        .map(TestResult::from)
        .map_err(to_py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(u_test, m)?)?;
    Ok(())
}
