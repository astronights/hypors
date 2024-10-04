use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::Normal;

/// Performs a paired two-sample Z-test on two related samples.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data.
/// * `series2` - A `Series` containing the second set of sample data.
/// * `pop_std_diff` - The population standard deviation of the differences between the samples.
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
/// Returns a `PolarsError` if the mean of differences cannot be computed or other calculations fail.
pub fn z_test_paired(
    series1: &Series,
    series2: &Series,
    pop_std_diff: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let diff_series = (series1 - series2).expect("Unable to get Series difference");

    let n = diff_series.len() as f64;
    let sample_mean_diff = diff_series
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean of differences".into()))?;
    let std_error = pop_std_diff / n.sqrt();

    let z_stat = sample_mean_diff / std_error;

    let z_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(z_stat, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(sample_mean_diff, std_error, alpha, &z_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
    };

    Ok(TestResult {
        test_statistic: z_stat,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}

/// Performs an independent two-sample Z-test on two unrelated samples.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data.
/// * `series2` - A `Series` containing the second set of sample data.
/// * `pop_std1` - The population standard deviation for the first sample.
/// * `pop_std2` - The population standard deviation for the second sample.
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
/// Returns a `PolarsError` if the means cannot be computed or other calculations fail.
pub fn z_test_ind(
    series1: &Series,
    series2: &Series,
    pop_std1: f64,
    pop_std2: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let mean1 = series1
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;
    let mean2 = series2
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;

    let n1 = series1.len() as f64;
    let n2 = series2.len() as f64;

    let std_error = ((pop_std1.powi(2) / n1) + (pop_std2.powi(2) / n2)).sqrt();

    let z_stat = (mean1 - mean2) / std_error;

    let z_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(z_stat, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(mean1 - mean2, std_error, alpha, &z_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
    };

    Ok(TestResult {
        test_statistic: z_stat,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}