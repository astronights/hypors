//! Z-tests for proportions.

use crate::convert::{to_f64_vec, to_py_err};
use crate::types::{TailType, TestResult};
use pyo3::prelude::*;

/// One-sample proportion z-test against a known population proportion.
#[pyfunction]
#[pyo3(signature = (data, pop_proportion, tail, alpha))]
fn z_test(
    data: &Bound<'_, PyAny>,
    pop_proportion: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data = to_f64_vec("data", data)?;
    hypors::proportion::z_test(data, pop_proportion, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Two-sample proportion z-test; `pooled` selects the pooled-variance form.
#[pyfunction]
#[pyo3(signature = (data1, data2, tail, alpha, pooled))]
fn z_test_ind(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::proportion::z_test_ind(data1, data2, tail.into(), alpha, pooled)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Sample size required to detect a difference between two proportions.
#[pyfunction]
#[pyo3(signature = (p1, p2, alpha, power))]
fn prop_sample_size(p1: f64, p2: f64, alpha: f64, power: f64) -> f64 {
    hypors::proportion::prop_sample_size(p1, p2, alpha, power)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(z_test, m)?)?;
    m.add_function(wrap_pyfunction!(z_test_ind, m)?)?;
    m.add_function(wrap_pyfunction!(prop_sample_size, m)?)?;
    Ok(())
}
