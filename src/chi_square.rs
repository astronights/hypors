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

    let chi_square_stat = (n - 1.0) * sample_variance / pop_variance;
    let df = n - 1.0;
    let chi_distribution = ChiSquared::new(df).unwrap();

    // Compute p-value based on tail type
    let p_value = calculate_p(chi_square_stat, tail.clone(), &chi_distribution);

    // Reject null hypothesis if p-value is smaller than alpha
    let reject_null = p_value < alpha;

    // Compute the confidence interval for the population variance
    let confidence_interval = calculate_chi2_ci(pop_variance, alpha, &chi_distribution);

    let alt_hypothesis = match tail {
        TailType::Two => format!("H₁: σ² ≠ {}", pop_variance),
        TailType::Left => format!("H₁: σ² < {}", pop_variance),
        TailType::Right => format!("H₁: σ² > {}", pop_variance),
    };

    Ok(TestResult {
        test_statistic: chi_square_stat,
        p_value,
        confidence_interval,
        null_hypothesis: format!("H₀: σ² = {}", pop_variance),
        alt_hypothesis,
        reject_null,
    })
}

/// Perform a Chi-Square Test for Independence (using a contingency table).
///
/// # Arguments
///
/// * `contingency_table` - A 2D Series array representing the contingency table.
/// * `alpha` - Significance level for the test.
///
/// # Returns
///
/// Returns `Result<TestResult, PolarsError>`.
pub fn independence(contingency_table: &[Vec<f64>], alpha: f64) -> Result<TestResult, PolarsError> {
    let num_rows = contingency_table.len();
    let num_cols = contingency_table[0].len();
    let total = contingency_table
        .iter()
        .flat_map(|row| row.iter())
        .sum::<f64>();

    let mut expected = vec![vec![0.0; num_cols]; num_rows];

    for i in 0..num_rows {
        for j in 0..num_cols {
            let row_sum: f64 = contingency_table[i].iter().sum();
            let col_sum: f64 = contingency_table.iter().map(|r| r[j]).sum();
            expected[i][j] = row_sum * col_sum / total;
        }
    }

    let chi_square_stat = contingency_table
        .iter()
        .enumerate()
        .fold(0.0, |sum, (i, row)| {
            row.iter()
                .enumerate()
                .fold(sum, |inner_sum, (j, &observed)| {
                    inner_sum + ((observed - expected[i][j]).powi(2) / expected[i][j])
                })
        });

    let df = (num_rows - 1) * (num_cols - 1);
    let chi_distribution = ChiSquared::new(df as f64).unwrap();
    let p_value = calculate_p(chi_square_stat, TailType::Right, &chi_distribution);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic: chi_square_stat,
        p_value,
        reject_null,
        null_hypothesis: "H0: Variables are independent".to_string(),
        alt_hypothesis: "H1: Variables are not independent".to_string(),
        confidence_interval: (std::f64::NAN, std::f64::NAN),
    })
}

/// Perform a Chi-Square Goodness of Fit Test.
///
/// # Arguments
///
/// * `observed` - The observed frequencies as a Series.
/// * `expected` - The expected frequencies as a Series.
/// * `alpha` - Significance level for the test.
///
/// # Returns
///
/// Returns `Result<TestResult, PolarsError>`.
pub fn goodness_of_fit(
    observed: &Series,
    expected: &Series,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let obs_iter = observed.f64()?.into_no_null_iter();
    let exp_iter = expected.f64()?.into_no_null_iter();

    let chi_square_stat = obs_iter
        .zip(exp_iter)
        .fold(0.0, |sum, (obs, exp)| sum + (obs - exp).powi(2) / exp);

    let df = observed.len() as f64 - 1.0;
    let chi_distribution = ChiSquared::new(df).unwrap();
    let p_value = calculate_p(chi_square_stat, TailType::Right, &chi_distribution);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic: chi_square_stat,
        p_value,
        reject_null,
        null_hypothesis: "H0: Observed distribution matches expected distribution".to_string(),
        alt_hypothesis: "H1: Observed distribution does not match expected distribution"
            .to_string(),
        confidence_interval: (std::f64::NAN, std::f64::NAN),
    })
}
