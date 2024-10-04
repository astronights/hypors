use crate::common::{calculate_chi2_ci, calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::ChiSquared;

/// Perform a Chi-Square Test for Variance.
///
/// This test evaluates whether the sample variance significantly differs from a specified population variance.
///
/// # Arguments
///
/// * `data` - A `Series` containing the sample data.
/// * `pop_variance` - The hypothesized population variance against which the sample variance is tested.
/// * `tail` - The tail type for the test (Left, Right, or Two-tailed).
/// * `alpha` - The significance level for the test (typically 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, PolarsError>`, where `TestResult` contains:
/// - `test_statistic`: The calculated Chi-Square test statistic for variance.
/// - `p_value`: The p-value associated with the test statistic.
/// - `confidence_interval`: The confidence interval for the population variance.
/// - `null_hypothesis`: The statement of the null hypothesis (H0: σ² = population variance).
/// - `alt_hypothesis`: The statement of the alternative hypothesis, indicating how the sample variance relates to the population variance.
/// - `reject_null`: A boolean indicating whether to reject the null hypothesis.
///
/// # Example
///
/// ```rust
/// use crate::chi_square::variance;
/// use polars::prelude::*;
///
/// // Sample data
/// let data = Series::new("data", vec![4.0, 5.0, 6.0, 7.0, 8.0]);
/// let pop_variance = 2.0; // Hypothesized population variance
/// let alpha = 0.05; // Significance level
///
/// // Perform Chi-Square Test for Variance
/// let result = variance(&data, pop_variance, TailType::Two, alpha).unwrap();
/// println!("Test Statistic: {}, p-value: {}", result.test_statistic, result.p_value);
/// ```
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
