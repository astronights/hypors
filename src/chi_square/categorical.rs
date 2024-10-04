use crate::common::{calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::ChiSquared;

/// Perform a Chi-Square Test for Independence using a contingency table.
///
/// This test evaluates whether there is a significant association between two categorical variables
/// based on a contingency table.
///
/// # Arguments
///
/// * `contingency_table` - A 2D vector of `f64` representing the observed frequencies in the contingency table.
/// * `alpha` - The significance level for the test (typically 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, PolarsError>`, where `TestResult` contains:
/// - `test_statistic`: The calculated Chi-Square test statistic.
/// - `p_value`: The p-value associated with the test statistic.
/// - `reject_null`: A boolean indicating whether to reject the null hypothesis.
/// - `null_hypothesis`: The statement of the null hypothesis (H0: Variables are independent).
/// - `alt_hypothesis`: The statement of the alternative hypothesis (Ha: Variables are not independent).
///
/// # Example
///
/// ```rust
/// use crate::chi_square::independence;
///
/// // Example contingency table
/// let contingency_table = vec![
///     vec![20.0, 30.0],
///     vec![50.0, 10.0],
/// ];
/// let alpha = 0.05;
///
/// // Perform Chi-Square Test for Independence
/// let result = independence(&contingency_table, alpha).unwrap();
/// println!("Test Statistic: {}, p-value: {}", result.test_statistic, result.p_value);
/// ```
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

    let test_statistic = contingency_table
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
    let p_value = calculate_p(test_statistic, TailType::Right, &chi_distribution);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval: (std::f64::NAN, std::f64::NAN),
        null_hypothesis: "H0: Variables are independent".to_string(),
        alt_hypothesis: "Ha: Variables are not independent".to_string(),
        reject_null,
    })
}

/// Perform a Chi-Square Goodness of Fit Test.
///
/// This test determines whether the observed frequencies differ significantly from expected frequencies.
///
/// # Arguments
///
/// * `observed` - A `Series` representing the observed frequencies.
/// * `expected` - A `Series` representing the expected frequencies.
/// * `alpha` - The significance level for the test (typically 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, PolarsError>`, where `TestResult` contains:
/// - `test_statistic`: The calculated Chi-Square test statistic.
/// - `p_value`: The p-value associated with the test statistic.
/// - `reject_null`: A boolean indicating whether to reject the null hypothesis.
/// - `null_hypothesis`: The statement of the null hypothesis (H0: Observed distribution matches expected distribution).
/// - `alt_hypothesis`: The statement of the alternative hypothesis (Ha: Observed distribution does not match expected distribution).
///
/// # Example
///
/// ```rust
/// use crate::chi_square::goodness_of_fit;
/// use polars::prelude::*;
///
/// // Observed and expected frequencies
/// let observed = Series::new("Observed", vec![30.0, 10.0, 20.0]);
/// let expected = Series::new("Expected", vec![25.0, 15.0, 20.0]);
/// let alpha = 0.05;
///
/// // Perform Chi-Square Goodness of Fit Test
/// let result = goodness_of_fit(&observed, &expected, alpha).unwrap();
/// println!("Test Statistic: {}, p-value: {}", result.test_statistic, result.p_value);
/// ```
pub fn goodness_of_fit(
    observed: &Series,
    expected: &Series,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let obs_iter = observed.f64()?.into_no_null_iter();
    let exp_iter = expected.f64()?.into_no_null_iter();

    let test_statistic = obs_iter
        .zip(exp_iter)
        .fold(0.0, |sum, (obs, exp)| sum + (obs - exp).powi(2) / exp);

    let df = observed.len() as f64 - 1.0;
    let chi_distribution = ChiSquared::new(df).unwrap();
    let p_value = calculate_p(test_statistic, TailType::Right, &chi_distribution);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval: (std::f64::NAN, std::f64::NAN),
        null_hypothesis: "H0: Observed distribution matches expected distribution".to_string(),
        alt_hypothesis: "Ha: Observed distribution does not match expected distribution"
            .to_string(),
        reject_null,
    })
}
