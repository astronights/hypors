use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::Normal;

/// This module implements the one-sample Z-test.
///
/// The one-sample Z-test evaluates whether the mean of a sample differs
/// from a known population mean, given the population standard deviation.
///
/// ## Function
///
/// - `z_test`: Conducts a one-sample Z-test on the provided data series.
///
/// ## Arguments
///
/// - `series`: A `Series` containing the sample data.
/// - `pop_mean`: The population mean to compare against.
/// - `pop_std`: The population standard deviation.
/// - `tail`: Specifies the type of tail (left, right, or two) for the hypothesis test.
/// - `alpha`: The significance level for the test (e.g., 0.05 for 95% confidence).
///
/// ## Returns
///
/// Returns a `TestResult` struct containing:
/// - `test_statistic`: The calculated Z statistic.
/// - `p_value`: The p-value associated with the test statistic.
/// - `confidence_interval`: The confidence interval for the sample mean.
/// - `null_hypothesis`: A string representing the null hypothesis.
/// - `alt_hypothesis`: A string representing the alternative hypothesis.
/// - `reject_null`: A boolean indicating whether to reject the null hypothesis.
///
/// ## Errors
///
/// Returns a `PolarsError` if there are issues calculating the mean or variance
/// of the series, or if the statistical calculations fail.
///
/// ## Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{z_test, TailType};
///
/// let series = Series::new("data".into(), &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let pop_mean = 2.0;
/// let pop_std = 1.0; // Known population standard deviation
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
///
/// // Perform the one-sample Z-test
/// let result = z_test(&series, pop_mean, pop_std, tail, alpha).unwrap();
///
/// // Check if the p-value is within a valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn z_test(
    series: &Series,
    pop_mean: f64,
    pop_std: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let sample_mean = series
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample mean".into()))?;

    let n = series.len() as f64;
    let std_error = pop_std / n.sqrt();

    let test_statistic = (sample_mean - pop_mean) / std_error;

    let z_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &z_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ >= {}", pop_mean),
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
