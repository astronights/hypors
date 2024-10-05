use pyo3::prelude::*;
use pyo3::types::{PyDict};
use serde::{Deserialize, Serialize};

#[pyclass(eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TailType {
    #[pyo3(name = "Left")]
    Left,
    #[pyo3(name = "Right")]
    Right,
    #[pyo3(name = "Two")]
    Two,
}

#[pymethods]
impl TailType {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            TailType::Left => "Left",
            TailType::Right => "Right",
            TailType::Two => "Two",
        }
    }
}

#[pyclass(eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestResult {
    test_statistic: f64,
    p_value: f64,
    confidence_interval: (f64, f64),
    null_hypothesis: String,
    alt_hypothesis: String,
    reject_null: bool,
}

#[pymethods]
impl TestResult {
    #[new]
    #[pyo3(signature = (
        test_statistic,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null
    ))]
    fn new(
        test_statistic: f64,
        p_value: f64,
        confidence_interval: (f64, f64),
        null_hypothesis: String,
        alt_hypothesis: String,
        reject_null: bool,
    ) -> Self {
        Self {
            test_statistic,
            p_value,
            confidence_interval,
            null_hypothesis,
            alt_hypothesis,
            reject_null,
        }
    }

    // Getters and setters
    #[getter]
    fn test_statistic(&self) -> f64 {
        self.test_statistic
    }

    #[setter]
    fn set_test_statistic(&mut self, value: f64) {
        self.test_statistic = value;
    }

    #[getter]
    fn p_value(&self) -> f64 {
        self.p_value
    }

    #[setter]
    fn set_p_value(&mut self, value: f64) {
        self.p_value = value;
    }

    #[getter]
    fn confidence_interval(&self) -> (f64, f64) {
        self.confidence_interval
    }

    #[setter]
    fn set_confidence_interval(&mut self, value: (f64, f64)) {
        self.confidence_interval = value;
    }

    #[getter]
    fn null_hypothesis(&self) -> &str {
        &self.null_hypothesis
    }

    #[setter]
    fn set_null_hypothesis(&mut self, value: String) {
        self.null_hypothesis = value;
    }

    #[getter]
    fn alt_hypothesis(&self) -> &str {
        &self.alt_hypothesis
    }

    #[setter]
    fn set_alt_hypothesis(&mut self, value: String) {
        self.alt_hypothesis = value;
    }

    #[getter]
    fn reject_null(&self) -> bool {
        self.reject_null
    }

    #[setter]
    fn set_reject_null(&mut self, value: bool) {
        self.reject_null = value;
    }

    // Method for serialization to a dictionary
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("test_statistic", self.test_statistic)?;
        dict.set_item("p_value", self.p_value)?;
        dict.set_item(
            "confidence_interval",
            vec![self.confidence_interval.0, self.confidence_interval.1],
        )?;
        dict.set_item("null_hypothesis", &self.null_hypothesis)?;
        dict.set_item("alt_hypothesis", &self.alt_hypothesis)?;
        dict.set_item("reject_null", self.reject_null)?;
        Ok(dict.into())
    }
}
