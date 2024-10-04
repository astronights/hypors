use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::Normal;

/// Performs a one-sample proportion test on the provided data series.
///
/// This test evaluates whether the sample proportion differs significantly from a specified population proportion.
///
/// # Arguments
///
/// * `series` - A `Series` containing the sample data, where values should be binary (1 for success, 0 for failure).
/// * `pop_proportion` - The hypothesized population proportion to test against (e.g., 0.5).
/// * `tail` - The type of tail for the test: `TailType::Left`, `TailType::Right`, or `TailType::Two`.
/// * `alpha` - The significance level for the test (e.g., 0.05 for a 95% confidence level).
///
/// # Returns
///
/// Returns a `Result<TestResult, PolarsError>`, where:
/// - `TestResult` includes the test statistic, p-value, confidence interval,
///   null and alternative hypotheses, and a boolean indicating if the null hypothesis is rejected.
///
/// # Errors
///
/// Returns a `PolarsError` if any calculations fail, such as when computing the sample proportion.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use hypors::{proportion::z_test, TailType};
///
/// let series = Series::new("data", &[1, 0, 1, 1, 0]);
/// let pop_proportion = 0.5;
/// let tail = TailType::Two;
/// let alpha = 0.05;
///
/// let result = z_test(&series, pop_proportion, tail, alpha).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert!(result.reject_null == (result.p_value < alpha));
/// ```
pub fn z_test(
    series: &Series,
    pop_proportion: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let n = series.len() as f64;
    let successes = series.sum::<u32>().unwrap() as f64; // Assuming series has binary values (0 and 1)
    let sample_proportion = successes / n;

    let std_error = (pop_proportion * (1.0 - pop_proportion) / n).sqrt();
    let test_statistic = (sample_proportion - pop_proportion) / std_error;

    let normal_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let p_value = calculate_p(test_statistic, tail.clone(), &normal_dist);
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
        TailType::Two => format!("Ha: p ≠ {}", pop_proportion),
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
