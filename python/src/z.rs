//! Z-tests for means with known population standard deviations.

use crate::convert::{to_f64_vec, to_py_err};
use crate::types::{TailType, TestResult};
use pyo3::prelude::*;

/// One-sample z-test against a known population mean and standard deviation.
#[pyfunction]
#[pyo3(signature = (data, pop_mean, pop_std, tail, alpha))]
fn z_test(
    data: &Bound<'_, PyAny>,
    pop_mean: f64,
    pop_std: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data = to_f64_vec("data", data)?;
    hypors::z::z_test(data, pop_mean, pop_std, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Paired z-test, given the population standard deviation of the differences.
#[pyfunction]
#[pyo3(signature = (data1, data2, pop_std_diff, tail, alpha))]
fn z_test_paired(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    pop_std_diff: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::z::z_test_paired(data1, data2, pop_std_diff, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Independent two-sample z-test with known population standard deviations.
#[pyfunction]
#[pyo3(signature = (data1, data2, pop_std1, pop_std2, tail, alpha))]
fn z_test_ind(
    data1: &Bound<'_, PyAny>,
    data2: &Bound<'_, PyAny>,
    pop_std1: f64,
    pop_std2: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data1 = to_f64_vec("data1", data1)?;
    let data2 = to_f64_vec("data2", data2)?;
    hypors::z::z_test_ind(data1, data2, pop_std1, pop_std2, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Sample size required for a z-test at the given effect size, alpha and power.
#[pyfunction]
#[pyo3(signature = (effect_size, alpha, power, std_dev, tail))]
fn z_sample_size(effect_size: f64, alpha: f64, power: f64, std_dev: f64, tail: TailType) -> f64 {
    hypors::z::z_sample_size(effect_size, alpha, power, std_dev, tail.into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(z_test, m)?)?;
    m.add_function(wrap_pyfunction!(z_test_paired, m)?)?;
    m.add_function(wrap_pyfunction!(z_test_ind, m)?)?;
    m.add_function(wrap_pyfunction!(z_sample_size, m)?)?;
    Ok(())
}
