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
///     let data1 = Series::new("group1".into(), vec![1.0, 2.0, 3.0, 4.0]);
///     let data2 = Series::new("group2".into(), vec![2.5, 3.5, 4.5]);
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
    let mut combined = Vec::with_capacity(data1.len() + data2.len());
    combined.extend(data1.f64()?.into_iter().flatten().map(|v| (v, 1)));
    combined.extend(data2.f64()?.into_iter().flatten().map(|v| (v, 2)));

    // Rank the data with tie handling (average rank for ties)
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut rank_values = vec![0.0; combined.len()];
    let mut i = 0;

    // Assign ranks (average ranks for tied values)
    while i < combined.len() {
        let start = i;
        let mut end = i;

        while end + 1 < combined.len() && combined[end + 1].0 == combined[start].0 {
            end += 1;
        }

        // Average rank for tied values
        let rank_avg = ((start + 1) + (end + 1)) as f64 / 2.0;
        for v in rank_values.iter_mut().take(end + 1).skip(start) {
            *v = rank_avg;
        }

        i = end + 1;
    }

    let mut rank_sum1 = 0.0;
    let mut rank_sum2 = 0.0;

    for (rank, group) in rank_values.iter().zip(combined.iter()) {
        if group.1 == 1 {
            rank_sum1 += rank;
        } else {
            rank_sum2 += rank;
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
