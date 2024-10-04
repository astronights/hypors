use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use crate::t::t_test;
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::StudentsT;


/// Performs a paired two-sample t-test on two related samples.
///
/// This function evaluates whether the means of two related groups differ from each other.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data.
/// * `series2` - A `Series` containing the second set of sample data.
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
/// Returns a `PolarsError` if there are issues calculateulating the mean or variance of the series.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{t_test_paired, TailType};
///
/// let series1 = Series::new("data1".into(), &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let series2 = Series::new("data2".into(), &[1.1, 2.0, 1.7, 2.3, 2.6]);
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
///
/// // Perform the paired two-sample t-test
/// let result = t_test_paired(&series1, &series2, tail, alpha).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn t_test_paired(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let diff_series = (series1 - series2).expect("Unable to get Series difference");

    let mut result = t_test(&diff_series, 0.0, tail.clone(), alpha)?;

    result.null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    result.alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
    };

    Ok(result)
}

/// Performs an independent two-sample t-test on two unrelated samples.
///
/// This function evaluates whether the means of two independent groups differ from each other.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data.
/// * `series2` - A `Series` containing the second set of sample data.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `pooled` - Whether to pool variances (true for a standard t-test, false for Welch's t-test).
///
/// # Returns
///
/// A `TestResult` struct containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and a boolean indicating whether the null hypothesis should be rejected.
///
/// # Errors
///
/// Returns a `PolarsError` if there are issues calculateulating the mean or variance of the series.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{t_test_ind, TailType};
///
/// let series1 = Series::new("data1".into(), &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let series2 = Series::new("data2".into(), &[1.1, 2.0, 1.7, 2.3, 2.6]);
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
/// let pooled = false; // Use Welch's t-test
///
/// // Perform the independent two-sample t-test
/// let result = t_test_ind(&series1, &series2, tail, alpha, pooled).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn t_test_ind(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> Result<TestResult, PolarsError> {
    let mean1 = series1
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;
    let mean2 = series2
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;
    let var1 = series1
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute variance".into()))?;
    let var2 = series2
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute variance".into()))?;

    let n1 = series1.len() as f64;
    let n2 = series2.len() as f64;

    let (std_error, df) = if pooled {
        let pooled_var = (((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0);
        let std_error = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
        let df = n1 + n2 - 2.0;
        (std_error, df)
    } else {
        let std_error = ((var1 / n1) + (var2 / n2)).sqrt();
        let df = ((var1 / n1) + (var2 / n2)).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
        (std_error, df)
    };

    let test_statistic = (mean1 - mean2) / std_error;
    let t_dist = StudentsT::new(0.0, 1.0, df).expect("Failed to create StudentsT distribution");

    let p_value = calculate_p(test_statistic, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(mean1 - mean2, std_error, alpha, &t_dist);

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
        test_statistic,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}
