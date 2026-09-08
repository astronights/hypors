//! Student's t-tests.

use crate::convert::{to_f64_vec, to_py_err};
use crate::types::{TailType, TestResult};
use pyo3::prelude::*;

/// One-sample t-test against a known population mean.
#[pyfunction]
#[pyo3(signature = (data, pop_mean, tail, alpha))]
fn t_test(
    data: &Bound<'_, PyAny>,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data = to_f64_vec("data", data)?;
    hypors::t::t_test(data, pop_mean, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Paired two-sample t-test.
#[pyfunction]
#[pyo3(signature = (data1, data2, tail, alpha))]
fn t_test_paired(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::t::t_test_paired(data1, data2, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Independent two-sample t-test; `pooled` selects the equal-variance form.
#[pyfunction]
#[pyo3(signature = (data1, data2, tail, alpha, pooled))]
fn t_test_ind(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::t::t_test_ind(data1, data2, tail.into(), alpha, pooled)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Sample size required for a t-test at the given effect size, alpha and power.
#[pyfunction]
#[pyo3(signature = (effect_size, alpha, power, std_dev, tail))]
fn t_sample_size(effect_size: f64, alpha: f64, power: f64, std_dev: f64, tail: TailType) -> f64 {
    hypors::t::t_sample_size(effect_size, alpha, power, std_dev, tail.into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(t_test, m)?)?;
    m.add_function(wrap_pyfunction!(t_test_paired, m)?)?;
    m.add_function(wrap_pyfunction!(t_test_ind, m)?)?;
    m.add_function(wrap_pyfunction!(t_sample_size, m)?)?;
    Ok(())
}
