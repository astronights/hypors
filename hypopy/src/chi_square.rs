//! Chi-square tests for categorical data and variance.

use crate::convert::{to_f64_rows, to_f64_vec, to_py_err, to_usize_vec};
use crate::types::{TailType, TestResult};
use pyo3::prelude::*;

/// Chi-square test of independence over a contingency table.
///
/// `contingency_table` is an iterable of rows, each an iterable of counts.
#[pyfunction]
#[pyo3(signature = (contingency_table, alpha))]
fn independence(contingency_table: &Bound<'_, PyAny>, alpha: f64) -> PyResult<TestResult> {
    let table = to_f64_rows("contingency_table", contingency_table)?;
    hypors::chi_square::independence(&table, alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Chi-square goodness-of-fit test comparing observed against expected counts.
#[pyfunction]
#[pyo3(signature = (observed, expected, alpha))]
fn goodness_of_fit(
    observed: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
    alpha: f64,
) -> PyResult<TestResult> {
    let observed = to_f64_vec("observed", observed)?;
    let expected = to_f64_vec("expected", expected)?;
    hypors::chi_square::goodness_of_fit(observed, expected, alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Chi-square test for a sample variance against a known population variance.
#[pyfunction]
#[pyo3(signature = (data, pop_variance, tail, alpha))]
fn variance(
    data: &Bound<'_, PyAny>,
    pop_variance: f64,
    tail: TailType,
    alpha: f64,
) -> PyResult<TestResult> {
    let data = to_f64_vec("data", data)?;
    hypors::chi_square::variance(data, pop_variance, tail.into(), alpha)
        .map(TestResult::from)
        .map_err(to_py_err)
}

/// Sample size for a goodness-of-fit test.
#[pyfunction]
#[pyo3(signature = (expected_counts, alpha))]
fn chi2_sample_size_gof(expected_counts: &Bound<'_, PyAny>, alpha: f64) -> PyResult<f64> {
    let counts = to_usize_vec("expected_counts", expected_counts)?;
    Ok(hypors::chi_square::chi2_sample_size_gof(&counts, alpha))
}

/// Sample size for a test of independence.
#[pyfunction]
#[pyo3(signature = (expected_counts, alpha))]
fn chi2_sample_size_ind(expected_counts: &Bound<'_, PyAny>, alpha: f64) -> PyResult<f64> {
    let counts = to_usize_vec("expected_counts", expected_counts)?;
    Ok(hypors::chi_square::chi2_sample_size_ind(&counts, alpha))
}

/// Sample size for a variance test.
#[pyfunction]
#[pyo3(signature = (effect_size, alpha, power, variance))]
fn chi2_sample_size_variance(effect_size: f64, alpha: f64, power: f64, variance: f64) -> f64 {
    hypors::chi_square::chi2_sample_size_variance(effect_size, alpha, power, variance)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(independence, m)?)?;
    m.add_function(wrap_pyfunction!(goodness_of_fit, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_sample_size_gof, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_sample_size_ind, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_sample_size_variance, m)?)?;
    Ok(())
}
