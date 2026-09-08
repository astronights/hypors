//! One-way analysis of variance.

use crate::convert::{to_f64_rows, to_py_err};
use crate::types::TestResult;
use pyo3::prelude::*;

/// One-way ANOVA across two or more independent groups.
///
/// `data_groups` is an iterable of groups, each itself an iterable of numbers.
#[pyfunction]
#[pyo3(signature = (data_groups, alpha))]
fn anova(data_groups: &Bound<'_, PyAny>, alpha: f64) -> PyResult<TestResult> {
    let groups = to_f64_rows("data_groups", data_groups)?;
    hypors::anova::anova(&groups, alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Sample size per group required for a one-way ANOVA.
#[pyfunction]
#[pyo3(signature = (effect_size, alpha, power, num_groups))]
fn f_sample_size(effect_size: f64, alpha: f64, power: f64, num_groups: usize) -> f64 {
    hypors::anova::f_sample_size(effect_size, alpha, power, num_groups)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anova, m)?)?;
    m.add_function(wrap_pyfunction!(f_sample_size, m)?)?;
    Ok(())
}
