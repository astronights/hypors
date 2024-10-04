use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::Normal;

/// Performs a one-sample Z-test on the provided data series.
///
/// # Arguments
///
/// * `series` - A `Series` containing the sample data.
/// * `pop_mean` - The population mean to test against.
/// * `pop_std` - The population standard deviation.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// A `TestResult` struct containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and a boolean indicating whether the null hypothesis should be rejected.
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
