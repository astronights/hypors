use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::StudentsT;

/// Performs a one-sample t-test on the provided data series.
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
/// use hypors::{one_sample, TailType};
///
/// let series = Series::new("data", &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let pop_mean = 2.0;
/// let tail = TailType::Two;
/// let alpha = 0.05;
///
/// let result = one_sample(&series, pop_mean, tail, alpha).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn one_sample(
    series: &Series,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let sample_mean = series
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample mean".into()))?;
    let sample_var = series
        .var(0)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample variance".into()))?;

    let n = series.len() as f64;
    let std_error = (sample_var / n).sqrt();

    let t_stat = (sample_mean - pop_mean) / std_error;
    let df = n - 1.0;

    let t_dist = StudentsT::new(0.0, 1.0, df).expect("Failed to create StudentsT distribution");

    let p_value = calculate_p(t_stat, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &t_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ >= {}", pop_mean),
        TailType::Right => format!("H0: µ <= {}", pop_mean),
        TailType::Two => format!("H0: µ = {}", pop_mean),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: µ < {}", pop_mean),
        TailType::Right => format!("Ha: µ > {}", pop_mean),
        TailType::Two => format!("Ha: µ != {}", pop_mean),
    };

    Ok(TestResult {
        test_statistic: t_stat,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}

/// Performs a paired two-sample t-test on two related samples.
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
/// Returns a `PolarsError` if there are issues calculating the mean or variance of the series.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{two_sample_paired, TailType};
///
/// let series1 = Series::new("data1", &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let series2 = Series::new("data2", &[1.1, 2.0, 1.7, 2.3, 2.6]);
/// let tail = TailType::Two;
/// let alpha = 0.05;
///
/// let result = two_sample_paired(&series1, &series2, tail, alpha).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn two_sample_paired(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let diff_series = (series1 - series2).expect("Unable to get Series difference");

    let mut result = one_sample(&diff_series, 0.0, tail.clone(), alpha)?;

    result.null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    result.alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 != µ2".to_string(),
    };

    Ok(result)
}

/// Performs an independent two-sample t-test on two unrelated samples.
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
/// Returns a `PolarsError` if there are issues calculating the mean or variance of the series.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{two_sample_ind, TailType};
///
/// let series1 = Series::new("data1", &[1.2, 2.3, 1.9, 2.5, 2.8]);
/// let series2 = Series::new("data2", &[1.1, 2.0, 1.7, 2.3, 2.6]);
/// let tail = TailType::Two;
/// let alpha = 0.05;
/// let pooled = false;  // Use Welch's t-test
///
/// let result = two_sample_ind(&series1, &series2, tail, alpha, pooled).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn two_sample_ind(
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
        .var(0)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute variance".into()))?;
    let var2 = series2
        .var(0)
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
        TailType::Two => "Ha: µ1 != µ2".to_string(),
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
