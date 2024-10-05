use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Represents the type of tail in hypothesis testing.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TailType {
    #[pyo3(name = "left")]
    Left,
    #[pyo3(name = "right")]
    Right,
    #[pyo3(name = "two")]
    Two,
}

/// Stores the result of a statistical test.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub null_hypothesis: String,
    pub alt_hypothesis: String,
    pub reject_null: bool,
}

#[pymethods]
impl TestResult {
    #[new]
    fn new(
        test_statistic: f64,
        p_value: f64,
        confidence_interval: (f64, f64),
        null_hypothesis: String,
        alt_hypothesis: String,
        reject_null: bool,
    ) -> Self {
        TestResult {
            test_statistic,
            p_value,
            confidence_interval,
            null_hypothesis,
            alt_hypothesis,
            reject_null,
        }
    }
}
