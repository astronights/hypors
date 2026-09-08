//! Conversions between Python objects and the plain Rust types the
//! `hypors` crate works with.
//!
//! The crate takes `IntoIterator<Item: Into<f64>>`, so anything iterable on
//! the Python side works: lists, tuples, generators, `numpy` arrays,
//! `pandas`/`polars` Series. Nothing here depends on those libraries — they
//! are simply iterated.

use hypors::common::StatError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Map the crate's error type onto the Python exception a caller expects:
/// bad input is a `ValueError`, a failed computation is a `RuntimeError`.
pub fn to_py_err(err: StatError) -> PyErr {
    match err {
        StatError::EmptyData => PyValueError::new_err("cannot perform test on empty data"),
        StatError::InsufficientData => {
            PyValueError::new_err("insufficient data for statistical test")
        }
        StatError::ComputeError(msg) => PyRuntimeError::new_err(msg),
    }
}

/// Collect any Python iterable of numbers into `Vec<f64>`.
///
/// `name` is the parameter being converted, so a bad element points at the
/// argument the caller passed rather than at an anonymous position.
pub fn to_f64_vec(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let iter = obj.try_iter().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an iterable of numbers, got {}",
            type_name(obj)
        ))
    })?;

    let mut out = Vec::new();
    for (i, item) in iter.enumerate() {
        let item = item?;
        let value = item.extract::<f64>().map_err(|_| {
            PyValueError::new_err(format!(
                "{name}[{i}] must be a number, got {}",
                type_name(&item)
            ))
        })?;
        out.push(value);
    }
    Ok(out)
}

/// Collect an iterable of iterables into `Vec<Vec<f64>>`, for the group and
/// contingency-table arguments.
pub fn to_f64_rows(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let iter = obj.try_iter().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an iterable of iterables of numbers, got {}",
            type_name(obj)
        ))
    })?;

    let mut out = Vec::new();
    for (i, row) in iter.enumerate() {
        out.push(to_f64_vec(&format!("{name}[{i}]"), &row?)?);
    }
    Ok(out)
}

/// Collect an iterable of non-negative integers into `Vec<usize>`, for the
/// expected-count arguments.
pub fn to_usize_vec(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let iter = obj.try_iter().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an iterable of non-negative integers, got {}",
            type_name(obj)
        ))
    })?;

    let mut out = Vec::new();
    for (i, item) in iter.enumerate() {
        let item = item?;
        let value = item.extract::<usize>().map_err(|_| {
            PyValueError::new_err(format!(
                "{name}[{i}] must be a non-negative integer, got {}",
                repr(&item)
            ))
        })?;
        out.push(value);
    }
    Ok(out)
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

fn repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| type_name(obj))
}
