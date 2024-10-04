use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::StudentsT;

/// Performs a one-sample t-test on the provided data series.
///
/// This function compares the mean of a sample to a known population mean to determine if there is a statistically significant difference.
///
/// # Arguments
///
/// * `series` - A `Series` containing the sample data.
/// * `pop_mean` - The population mean to test against.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// A `TestResult` struct containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and a boolean indicating whether the null hypothesis should be rejected.
///
/// # Errors
///
/// Returns a `PolarsError` if there are issues calculating the mean or variance of the series.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{t_test, TailType};
///
/// let series = Series::new("data", &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let pop_mean = 2.0;
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
///
/// // Perform the one-sample t-test
/// let result = t_test(&series, pop_mean, tail, alpha).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn t_test(
    series: &Series,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let sample_mean = series
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample mean".into()))?;
    let sample_var = series
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample variance".into()))?;

    let n = series.len() as f64;
    let std_error = (sample_var / n).sqrt();

    let test_statistic = (sample_mean - pop_mean) / std_error;
    let df = n - 1.0;

    let t_dist = StudentsT::new(0.0, 1.0, df).expect("Failed to create StudentsT distribution");

    let p_value = calculate_p(test_statistic, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &t_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ >= {:1}", pop_mean),
        TailType::Right => format!("H0: µ <= {}", pop_mean),
        TailType::Two => format!("H0: µ = {}", pop_mean),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: µ < {}", pop_mean),
        TailType::Right => format!("Ha: µ > {}", pop_mean),
        TailType::Two => format!("Ha: µ ≠ {}", pop_mean),
    };

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}
