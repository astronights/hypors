use crate::common::{StatError, TailType, TestResult, calculate_p};
use statrs::distribution::ChiSquared;
use std::f64;

/// Perform a Chi-Square Test for Independence using a contingency table.
///
/// This test evaluates whether there is a significant association between two categorical variables
/// based on observed frequencies in a contingency table.
///
/// # Arguments
///
/// * `contingency_table` - A slice of row vectors (`Vec<Vec<f64>>`) representing the observed frequencies.
/// * `alpha` - The significance level for the test (commonly 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, StatError>`, where:
/// - `TestResult` contains:
///     - `test_statistic`: The calculated chi-square statistic.
///     - `p_value`: The p-value associated with the statistic.
///     - `reject_null`: Whether the null hypothesis is rejected.
///     - `null_hypothesis`: "H0: Variables are independent".
///     - `alt_hypothesis`: "Ha: Variables are not independent".
///     - `confidence_interval`: Not applicable; returns `(NaN, NaN)`.
///
/// # Errors
/// Returns `StatError` if:
/// - Input rows are unequal or contain fewer than 2 rows/columns.
/// - Frequencies are invalid (e.g., zero total).
///
/// # Example
///
/// ```rust
/// use hypors::chi_square::independence;
///
/// let table = vec![
///     vec![20.0, 30.0],
///     vec![50.0, 10.0],
/// ];
/// let alpha = 0.05;
///
/// let result = independence(&table, alpha).unwrap();
/// println!("Chi-square: {}", result.test_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Reject null: {}", result.reject_null);
/// ```
pub fn independence(contingency_table: &[Vec<f64>], alpha: f64) -> Result<TestResult, StatError> {
    let num_rows = contingency_table.len();
    if num_rows < 2 {
        return Err(StatError::ComputeError("At least two rows required".into()));
    }

    let num_cols = contingency_table[0].len();
    if num_cols < 2 || !contingency_table.iter().all(|row| row.len() == num_cols) {
        return Err(StatError::ComputeError(
            "All rows must have equal and ≥2 columns".into(),
        ));
    }

    let total: f64 = contingency_table.iter().flatten().sum();
    if total == 0.0 {
        return Err(StatError::ComputeError(
            "Total frequency must be greater than zero".into(),
        ));
    }

    let mut expected = vec![vec![0.0; num_cols]; num_rows];
    let row_totals: Vec<f64> = contingency_table
        .iter()
        .map(|row| row.iter().sum())
        .collect();
    let col_totals: Vec<f64> = (0..num_cols)
        .map(|j| contingency_table.iter().map(|row| row[j]).sum())
        .collect();

    for i in 0..num_rows {
        for j in 0..num_cols {
            expected[i][j] = row_totals[i] * col_totals[j] / total;
        }
    }

    let test_statistic = contingency_table
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, &obs)| {
                    let exp = expected[i][j];
                    if exp == 0.0 {
                        0.0
                    } else {
                        (obs - exp).powi(2) / exp
                    }
                })
                .sum::<f64>()
        })
        .sum::<f64>();

    let df = (num_rows - 1) * (num_cols - 1);
    let chi_distribution = ChiSquared::new(df as f64)
        .map_err(|e| StatError::ComputeError(format!("Chi-squared distribution error: {e}")))?;
    let p_value = calculate_p(test_statistic, TailType::Right, &chi_distribution);
    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval: (f64::NAN, f64::NAN),
        null_hypothesis: "H0: Variables are independent".into(),
        alt_hypothesis: "Ha: Variables are not independent".into(),
        reject_null,
    })
}

/// Perform a Chi-Square Goodness of Fit Test.
///
/// This test evaluates whether an observed frequency distribution matches an expected distribution.
///
/// # Arguments
///
/// * `observed` - An iterator of observed frequencies.
/// * `expected` - An iterator of expected frequencies (must be same length as `observed`).
/// * `alpha` - Significance level (commonly 0.05).
///
/// # Returns
///
/// Returns a `Result<TestResult, StatError>`, where:
/// - `TestResult` contains:
///     - `test_statistic`: The calculated chi-square statistic.
///     - `p_value`: The p-value associated with the statistic.
///     - `reject_null`: Whether the null hypothesis is rejected.
///     - `null_hypothesis`: "H0: Observed distribution matches expected distribution".
///     - `alt_hypothesis`: "Ha: Observed distribution does not match expected distribution".
///     - `confidence_interval`: Not applicable; returns `(NaN, NaN)`.
///
/// # Errors
/// Returns `StatError` if:
/// - Inputs have different lengths or contain fewer than two categories.
/// - Invalid values are detected (e.g., expected = 0).
///
/// # Example
///
/// ```rust
/// use hypors::chi_square::goodness_of_fit;
///
/// let observed = vec![30.0, 10.0, 20.0];
/// let expected = vec![25.0, 15.0, 20.0];
/// let alpha = 0.05;
///
/// let result = goodness_of_fit(observed.iter().copied(), expected.iter().copied(), alpha).unwrap();
/// println!("Chi-square: {}", result.test_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Reject null: {}", result.reject_null);
/// ```
pub fn goodness_of_fit<O, E, T, U>(
    observed: O,
    expected: E,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    O: IntoIterator<Item = T>,
    E: IntoIterator<Item = U>,
    T: Into<f64>,
    U: Into<f64>,
{
    let observed: Vec<f64> = observed.into_iter().map(|x| x.into()).collect();
    let expected: Vec<f64> = expected.into_iter().map(|x| x.into()).collect();

    if observed.len() != expected.len() {
        return Err(StatError::ComputeError(
            "Observed and expected lengths must match".into(),
        ));
    }
    if observed.len() < 2 {
        return Err(StatError::ComputeError(
            "At least two categories required".into(),
        ));
    }

    let test_statistic: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&obs, &exp)| {
            if exp == 0.0 {
                0.0
            } else {
                (obs - exp).powi(2) / exp
            }
        })
        .sum();

    let df = (observed.len() - 1) as f64;
    let chi_distribution = ChiSquared::new(df)
        .map_err(|e| StatError::ComputeError(format!("Chi-squared distribution error: {e}")))?;
    let p_value = calculate_p(test_statistic, TailType::Right, &chi_distribution);
    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic,
        p_value,
        confidence_interval: (f64::NAN, f64::NAN),
        null_hypothesis: "H0: Observed distribution matches expected distribution".into(),
        alt_hypothesis: "Ha: Observed distribution does not match expected distribution".into(),
        reject_null,
    })
}
