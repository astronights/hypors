use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::Normal;

/// Performs an independent two-sample proportion test on two unrelated samples.
///
/// This test assesses whether the proportions of successes in two independent samples are significantly different.
///
/// # Arguments
///
/// * `series1` - A `Series` containing the first set of sample data, where values are binary (1 for success, 0 for failure).
/// * `series2` - A `Series` containing the second set of sample data, also binary (1 for success, 0 for failure).
/// * `tail` - Specifies the type of tail for the test: `TailType::Left`, `TailType::Right`, or `TailType::Two`.
/// * `alpha` - The significance level for the test (e.g., 0.05 for a 95% confidence level).
/// * `pooled` - A boolean indicating whether to use pooled proportions for calculating the standard error.
///
/// # Returns
///
/// Returns a `Result<TestResult, PolarsError>`, where:
/// - `TestResult` contains the test statistic, p-value, confidence interval,
///   null and alternative hypotheses, and a boolean indicating if the null hypothesis is rejected.
///
/// # Errors
///
/// Returns a `PolarsError` if any calculations fail, such as when computing the sample proportions.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{proportion::z_test_ind, TailType};
///
/// let series1 = Series::new("data1".into(), &[1, 0, 1, 1, 0]);
/// let series2 = Series::new("data2".into(), &[0, 0, 1, 1, 1]);
/// let tail = TailType::Two;
/// let alpha = 0.05;
/// let pooled = true; // Use pooled proportions
///
/// let result = z_test_ind(&series1, &series2, tail, alpha, pooled).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn z_test_ind(
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

    let test_statistic = (p1 - p2) / std_error;
    let normal_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(test_statistic, tail.clone(), &normal_dist);
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
        TailType::Two => "Ha: p1 ≠ p2".to_string(),
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
