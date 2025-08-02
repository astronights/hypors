use crate::common::{TailType, TestResult, calculate_chi2_ci, calculate_p};
use statrs::distribution::ChiSquared;
use std::f64;

/// Perform a Chi-Square Test for Variance.
///
/// This test evaluates whether the sample variance significantly differs from a specified population variance.
///
/// # Arguments
///
/// * `data` - An iterator of numeric values convertible to `f64`.
/// * `pop_variance` - The hypothesized population variance (σ²).
/// * `tail` - The direction of the test: `TailType::Left`, `Right`, or `Two`.
/// * `alpha` - The significance level (e.g., 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, String>`, where `TestResult` contains:
/// - `test_statistic`: The calculated Chi-Square test statistic.
/// - `p_value`: The p-value associated with the test statistic.
/// - `confidence_interval`: The confidence interval for the population variance.
/// - `null_hypothesis`: The null hypothesis statement (H₀: σ² = population variance).
/// - `alt_hypothesis`: The alternative hypothesis depending on the tail.
/// - `reject_null`: A boolean indicating whether to reject the null hypothesis.
///
/// # Example
///
/// ```rust
/// use hypors::chi_square::variance;
/// use hypors::common::TailType;
///
/// let data = vec![4.0, 5.0, 6.0, 7.0, 8.0];
/// let pop_variance = 2.0;
/// let alpha = 0.05;
///
/// let result = variance(data.iter().copied(), pop_variance, TailType::Two, alpha).unwrap();
/// println!("Chi-Square Statistic: {}", result.test_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Reject Null Hypothesis: {}", result.reject_null);
/// ```
pub fn variance<I, T>(
    data: I,
    pop_variance: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, String>
where
    I: IntoIterator<Item = T>,
    T: Into<f64>,
{
    // Collect data into Vec<f64>
    let sample_data: Vec<f64> = data.into_iter().map(|x| x.into()).collect();

    let n = sample_data.len();

    if n < 2 {
        return Err("Sample size must be at least 2.".to_string());
    }

    if !pop_variance.is_finite() || pop_variance <= 0.0 {
        return Err("Population variance must be a positive finite number.".to_string());
    }

    let mean = sample_data.iter().sum::<f64>() / n as f64;
    let sample_variance =
        sample_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);

    let test_statistic = (n as f64 - 1.0) * sample_variance / pop_variance;
    let df = n as f64 - 1.0;
    let chi_distribution = ChiSquared::new(df).map_err(|e| format!("Chi-squared error: {e}"))?;

    let p_value = calculate_p(test_statistic, tail.clone(), &chi_distribution);
    let reject_null = p_value < alpha;

    let confidence_interval = calculate_chi2_ci(pop_variance, alpha, &chi_distribution);

    let alt_hypothesis = match tail {
        TailType::Two => format!("Ha: σ² ≠ {pop_variance}"),
        TailType::Left => format!("Ha: σ² < {pop_variance}"),
        TailType::Right => format!("Ha: σ² > {pop_variance}"),
    };

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval,
        null_hypothesis: format!("H0: σ² = {pop_variance}"),
        alt_hypothesis,
        reject_null,
    })
}
