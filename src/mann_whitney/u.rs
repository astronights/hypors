use crate::common::{calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::Normal;
use std::f64;

/// Perform the Mann-Whitney U Test for comparing two independent samples.
///
/// This test evaluates whether the distributions of two independent groups are 
/// equal by ranking all observations and comparing the sum of ranks for each group.
///
/// # Arguments
///
/// * `data1` - A Series containing the data for the first group.
/// * `data2` - A Series containing the data for the second group.
/// * `alpha` - The significance level for the test, typically set at 0.05.
/// * `tail_type` - The type of tail for the test, which can be one of the following:
///   - `TailType::Left`: Test for the first group having a smaller distribution.
///   - `TailType::Right`: Test for the first group having a larger distribution.
///   - `TailType::Two`: Test for equality of distributions (two-tailed).
///
/// # Returns
///
/// Returns `Result<TestResult, PolarsError>`, where `TestResult` contains:
/// - `test_statistic`: The computed U statistic.
/// - `p_value`: The p-value for the test.
/// - `confidence_interval`: The confidence interval for the median difference (not applicable for U test).
/// - `null_hypothesis`: The null hypothesis statement.
/// - `alt_hypothesis`: The alternative hypothesis statement.
/// - `reject_null`: Boolean indicating whether to reject the null hypothesis.
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use crate::mann_whitney::u_test;
///
/// fn main() -> Result<(), PolarsError> {
///     let data1 = Series::new("group1", vec![1.0, 2.0, 3.0, 4.0]);
///     let data2 = Series::new("group2", vec![2.5, 3.5, 4.5]);
///     let alpha = 0.05;
///
///     let result = u_test(&data1, &data2, alpha, TailType::Two)?;
///
///     println!("U Statistic: {}", result.test_statistic);
///     println!("P-value: {}", result.p_value);
///     println!("Reject Null: {}", result.reject_null);
///     Ok(())
/// }
/// ```
pub fn u_test(
    data1: &Series,
    data2: &Series,
    alpha: f64,
    tail_type: TailType,
) -> Result<TestResult, PolarsError> {
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;

    // Combine the data
    let mut combined = Vec::new();
    for value in data1.f64()?.into_iter() {
        if let Some(v) = value {
            combined.push(v);
        }
    }
    for value in data2.f64()?.into_iter() {
        if let Some(v) = value {
            combined.push(v);
        }
    }

    // Rank the data
    let mut ranks: Vec<(f64, usize)> = combined.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    ranks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut rank_sum1 = 0.0;
    let mut rank_sum2 = 0.0;

    for (rank, (_value, original_index)) in ranks.iter().enumerate() {
        if original_index < &(n1 as usize) {
            rank_sum1 += (rank as f64) + 1.0; // Ranks start at 1
        } else {
            rank_sum2 += (rank as f64) + 1.0;
        }
    }

    // Calculate U statistic
    let u1 = rank_sum1 - (n1 * (n1 + 1.0) / 2.0);
    let u2 = rank_sum2 - (n2 * (n2 + 1.0) / 2.0);
    let u_statistic = u1.min(u2);

    // Calculate p-value (based on tail type)
    let total = n1 + n2;
    let mean_u = (n1 * n2) / 2.0;
    let variance_u = (n1 * n2 * (total + 1.0)) / 12.0;

    let z = (u_statistic - mean_u) / variance_u.sqrt();

    let dist = Normal::new(0.0, 1.0).unwrap();

    let p_value = calculate_p(z, tail_type, &dist);

    // Determine whether to reject the null hypothesis
    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic: u_statistic,
        p_value,
        confidence_interval: (f64::NAN, f64::NAN), // Confidence interval not applicable
        null_hypothesis: "H0: The distributions of both groups are equal.".to_string(),
        alt_hypothesis: "Ha: The distributions of both groups are not equal.".to_string(),
        reject_null,
    })
}
