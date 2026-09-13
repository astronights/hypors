//! The two types crossing the boundary in both directions.

use hypors::common::{TailType as RsTailType, TestResult as RsTestResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The arguments `TestResult.__new__` takes, in order.
type TestResultArgs = (f64, f64, (f64, f64), String, String, bool);

/// Which tail of the distribution a test is run against.
#[pyclass(eq, eq_int, frozen, from_py_object, module = "hypors.common")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TailType {
    /// Test whether the statistic is smaller than expected.
    Left,
    /// Test whether the statistic is larger than expected.
    Right,
    /// Test for a difference in either direction.
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

    fn __repr__(&self) -> String {
        format!("TailType.{}", self.name())
    }

    /// Rebuild a variant from its name, so the type can be pickled.
    #[staticmethod]
    fn _from_name(name: &str) -> PyResult<TailType> {
        match name {
            "Left" => Ok(TailType::Left),
            "Right" => Ok(TailType::Right),
            "Two" => Ok(TailType::Two),
            other => Err(PyValueError::new_err(format!("unknown TailType: {other}"))),
        }
    }

    // Without this the type cannot be pickled or copied, which breaks passing
    // a tail across a multiprocessing pool.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (String,))> {
        let from_name = slf.as_any().get_type().getattr("_from_name")?.unbind();
        Ok((from_name, (slf.get().name().to_string(),)))
    }
}

impl From<TailType> for RsTailType {
    fn from(tail: TailType) -> Self {
        match tail {
            TailType::Left => RsTailType::Left,
            TailType::Right => RsTailType::Right,
            TailType::Two => RsTailType::Two,
        }
    }
}

/// The outcome of a hypothesis test.
///
/// Attributes are read-only: a result describes a computation that already
/// happened, so mutating it would only ever misrepresent it.
#[pyclass(eq, frozen, skip_from_py_object, module = "hypors.common")]
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    /// The computed test statistic.
    #[pyo3(get)]
    pub test_statistic: f64,
    /// The p-value for the test.
    #[pyo3(get)]
    pub p_value: f64,
    /// Confidence interval, or `(nan, nan)` where the test does not define one.
    #[pyo3(get)]
    pub confidence_interval: (f64, f64),
    /// The null hypothesis being tested.
    #[pyo3(get)]
    pub null_hypothesis: String,
    /// The alternative hypothesis.
    #[pyo3(get)]
    pub alt_hypothesis: String,
    /// Whether the null hypothesis is rejected at the given alpha.
    #[pyo3(get)]
    pub reject_null: bool,
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
        reject_null,
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

    /// Return the result as a plain dict, for serialisation or tabulating.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("test_statistic", self.test_statistic)?;
        dict.set_item("p_value", self.p_value)?;
        dict.set_item(
            "confidence_interval",
            vec![self.confidence_interval.0, self.confidence_interval.1],
        )?;
        dict.set_item("null_hypothesis", &self.null_hypothesis)?;
        dict.set_item("alt_hypothesis", &self.alt_hypothesis)?;
        dict.set_item("reject_null", self.reject_null)?;
        Ok(dict)
    }

    // A result is exactly the sort of value that gets cached to disk or sent
    // between processes, so it has to survive pickle and copy.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, TestResultArgs)> {
        let cls = slf.as_any().get_type().unbind().into_any();
        let r = slf.get();
        Ok((
            cls,
            (
                r.test_statistic,
                r.p_value,
                r.confidence_interval,
                r.null_hypothesis.clone(),
                r.alt_hypothesis.clone(),
                r.reject_null,
            ),
        ))
    }

    fn __repr__(&self) -> String {
        // Python spelling of the bool: a repr should be readable as Python.
        format!(
            "TestResult(test_statistic={}, p_value={}, reject_null={})",
            self.test_statistic,
            self.p_value,
            if self.reject_null { "True" } else { "False" }
        )
    }
}

impl From<RsTestResult> for TestResult {
    fn from(r: RsTestResult) -> Self {
        Self {
            test_statistic: r.test_statistic,
            p_value: r.p_value,
            confidence_interval: r.confidence_interval,
            null_hypothesis: r.null_hypothesis,
            alt_hypothesis: r.alt_hypothesis,
            reject_null: r.reject_null,
        }
    }
}
