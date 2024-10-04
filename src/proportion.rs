use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::Normal;

/// Performs a one-sample proportion test on the provided data series.
///
/// # Arguments
///
/// * `series` - A `Series` containing the sample data (1 for success, 0 for failure).
/// * `pop_proportion` - The population proportion to test against (e.g., 0.5).
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
/// Returns a `PolarsError` if there are issues calculating the sample proportion.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{one_sample_proportion, TailType};
///
/// let series = Series::new("data", &[1, 0, 1, 1, 0]);
/// let pop_proportion = 0.5;
/// let tail = TailType::Two;
/// let alpha = 0.05;
///
/// let result = one_sample(&series, pop_proportion, tail, alpha).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn one_sample(
    series: &Series,
    pop_proportion: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let n = series.len() as f64;
    let successes = series.sum::<u32>().unwrap() as f64; // Assuming series has binary values (0 and 1)
    let sample_proportion = successes / n;

    let std_error = (pop_proportion * (1.0 - pop_proportion) / n).sqrt();
    let z_stat = (sample_proportion - pop_proportion) / std_error;

    let normal_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(z_stat, tail.clone(), &normal_dist);
    let confidence_interval = calculate_ci(sample_proportion, std_error, alpha, &normal_dist);
    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: p >= {}", pop_proportion),
        TailType::Right => format!("H0: p <= {}", pop_proportion),
        TailType::Two => format!("H0: p = {}", pop_proportion),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: p < {}", pop_proportion),
        TailType::Right => format!("Ha: p > {}", pop_proportion),
        TailType::Two => format!("Ha: p != {}", pop_proportion),
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

/// Performs an independent two-sample proportion test on two unrelated samples.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data (1 for success, 0 for failure).
/// * `series2` - A `Series` containing the second set of sample data (1 for success, 0 for failure).
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `pooled` - Whether to use pooled proportions for the standard error calculation.
///
/// # Returns
///
/// A `TestResult` struct containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and a boolean indicating whether the null hypothesis should be rejected.
///
/// # Errors
///
/// Returns a `PolarsError` if there are issues calculating the sample proportions.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{two_sample, TailType};
///
/// let series1 = Series::new("data1", &[1, 0, 1, 1, 0]);
/// let series2 = Series::new("data2", &[0, 0, 1, 1, 1]);
/// let tail = TailType::Two;
/// let alpha = 0.05;
/// let pooled = true; // Use pooled proportions
///
/// let result = two_sample(&series1, &series2, tail, alpha, pooled).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn two_sample(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> Result<TestResult, PolarsError> {
    let n1 = series1.len() as f64;
    let n2 = series2.len() as f64;

    let successes1 = series1.sum::<u32>().unwrap() as f64;
    let successes2 = series2.sum::<u32>().unwrap() as f64;

    let p1 = successes1 / n1; // Sample proportion for the first group
    let p2 = successes2 / n2; // Sample proportion for the second group

    let std_error = if pooled {
        let pooled_proportion = (successes1 + successes2) / (n1 + n2);
        (pooled_proportion * (1.0 - pooled_proportion) * (1.0 / n1 + 1.0 / n2)).sqrt()
    } else {
        ((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2)).sqrt()
    };

    let z_stat = (p1 - p2) / std_error;
    let normal_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(z_stat, tail.clone(), &normal_dist);
    let confidence_interval = calculate_ci(p1 - p2, std_error, alpha, &normal_dist);
    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: p1 >= p2".to_string(),
        TailType::Right => "H0: p1 <= p2".to_string(),
        TailType::Two => "H0: p1 = p2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: p1 < p2".to_string(),
        TailType::Right => "Ha: p1 > p2".to_string(),
        TailType::Two => "Ha: p1 != p2".to_string(),
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
