use crate::common::{calculate_chi2_ci, calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::ChiSquared;

/// Perform a Chi-Square Test for Variance.
///
/// # Arguments
///
/// * `data` - The data Series.
/// * `pop_variance` - The hypothesized population variance.
/// * `alpha` - Significance level for the test.
/// * `tail` - Tail type (left, right, or two-tailed).
///
/// # Returns
///
/// Returns `Result<TestResult, PolarsError>`, with the confidence interval for the variance.
pub fn variance(
    data: &Series,
    pop_variance: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let n = data.len() as f64;
    let sample_variance = data
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample variance".into()))?;

    let test_statistic = (n - 1.0) * sample_variance / pop_variance;
    let df = n - 1.0;
    let chi_distribution = ChiSquared::new(df).unwrap();

    // Compute p-value based on tail type
    let p_value = calculate_p(test_statistic, tail.clone(), &chi_distribution);

    // Reject null hypothesis if p-value is smaller than alpha
    let reject_null = p_value < alpha;

    // Compute the confidence interval for the population variance
    let confidence_interval = calculate_chi2_ci(pop_variance, alpha, &chi_distribution);

    let alt_hypothesis = match tail {
        TailType::Two => format!("Ha: σ² ≠ {}", pop_variance),
        TailType::Left => format!("Ha: σ² < {}", pop_variance),
        TailType::Right => format!("Ha: σ² > {}", pop_variance),
    };

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval,
        null_hypothesis: format!("H0: σ² = {}", pop_variance),
        alt_hypothesis,
        reject_null,
    })
}