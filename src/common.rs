use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

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
/// use hypors::TestResult;
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

/// Calculates the p-value for a given test statistic.
///
/// # Arguments
///
/// * `t_stat` - The test statistic (e.g., t-statistic).
/// * `tail` - The type of tail (left, right, or two).
/// * `dist` - The statistical distribution to be used, which must implement the `ContinuousCDF` trait.
///
/// # Returns
///
/// The p-value corresponding to the test statistic and tail type.
///
/// # Example
///
/// ```rust
/// use statrs::distribution::{StudentsT, ContinuousCDF};
/// use hypors::TailType;
/// use hypors::calculate_p;
///
/// let t_stat = 2.0;
/// let tail = TailType::Two;
/// let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();  // Student's t-distribution with 10 degrees of freedom
///
/// let p_value = calculate_p(t_stat, tail, &t_dist);
/// assert!(p_value > 0.0 && p_value < 1.0);
/// ```
pub fn calculate_p(t_stat: f64, tail: TailType, dist: &dyn ContinuousCDF<f64, f64>) -> f64 {
    match tail {
        TailType::Left => dist.cdf(t_stat),
        TailType::Right => 1.0 - dist.cdf(t_stat),
        TailType::Two => 2.0 * (1.0 - dist.cdf(t_stat.abs())),
    }
}

/// Calculates the confidence interval for a sample mean.
///
/// # Arguments
///
/// * `sample_mean` - The sample mean for the dataset.
/// * `std_error` - The standard error of the mean.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `dist` - The statistical distribution to be used, which must implement the `ContinuousCDF` trait.
///
/// # Returns
///
/// A tuple `(lower_bound, upper_bound)` representing the confidence interval.
///
/// # Example
/// ```rust
/// use statrs::distribution::{StudentsT, ContinuousCDF};
/// use hypors::calculate_ci;
///
/// let sample_mean = 5.0;
/// let std_error = 1.5;
/// let alpha = 0.05;
/// let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();  // Student's t-distribution with 10 degrees of freedom
///
/// let ci = calculate_ci(sample_mean, std_error, alpha, &t_dist);
/// assert!(ci.0 < sample_mean && ci.1 > sample_mean);  // Lower and upper bounds should surround the mean
/// ```
pub fn calculate_ci(
    sample_mean: f64,
    std_error: f64,
    alpha: f64,
    dist: &dyn ContinuousCDF<f64, f64>,
) -> (f64, f64) {
    let margin_of_error = dist.inverse_cdf(1.0 - alpha / 2.0) * std_error;
    (sample_mean - margin_of_error, sample_mean + margin_of_error)
}

/// Calculates the confidence interval for Chi-squared distribution.
pub fn calculate_chi2_ci(sample_variance: f64, alpha: f64, dist: &ChiSquared) -> (f64, f64) {
    let df = dist.shape(); // Degrees of freedom
    let chi_square_lower = dist.inverse_cdf(alpha / 2.0);
    let chi_square_upper = dist.inverse_cdf(1.0 - alpha / 2.0);

    // Confidence interval for variance: (n-1) * sample_variance / chi_square_stat
    let lower_bound = (df * sample_variance) / chi_square_upper;
    let upper_bound = (df * sample_variance) / chi_square_lower;
    (lower_bound, upper_bound)
}
