use serde::{Deserialize, Serialize};

/// Represents the type of tail in hypothesis testing.
#[derive(Debug, Clone, PartialEq)]
pub enum TailType {
    /// Left tail test (used for testing if the observed statistic is less than a critical value).
    Left,
    /// Right tail test (used for testing if the observed statistic is greater than a critical value).
    Right,
    /// Two tail test (used for testing if the observed statistic differs from the critical value in either direction).
    Two,
}

/// Stores the result of a statistical test, including test statistic, p-value, confidence interval,
/// and hypothesis testing information.
///
/// # Fields
///
/// * `test_statistic` - The value of the test statistic.
/// * `p_value` - The p-value associated with the test statistic.
/// * `confidence_interval` - The confidence interval for the estimate (lower, upper bounds).
/// * `null_hypothesis` - The null hypothesis being tested.
/// * `alt_hypothesis` - The alternative hypothesis being tested.
/// * `reject_null` - A boolean indicating whether the null hypothesis should be rejected.
///
/// # Example
///
/// ```rust
/// use hypors::common::TestResult;
///
/// let test_result = TestResult {
///     test_statistic: 2.5,
///     p_value: 0.02,
///     confidence_interval: (1.0, 3.0),
///     null_hypothesis: String::from("Mean equals 0"),
///     alt_hypothesis: String::from("Mean is not equal to 0"),
///     reject_null: true,
/// };
///
/// assert_eq!(test_result.test_statistic, 2.5);
/// assert_eq!(test_result.p_value, 0.02);
/// assert!(test_result.reject_null);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub null_hypothesis: String,
    pub alt_hypothesis: String,
    pub reject_null: bool,
}
