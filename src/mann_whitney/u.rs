use crate::common::{TailType, TestResult, calculate_p};
use statrs::distribution::Normal;

/// Perform the Mann-Whitney U Test for comparing two independent samples.
///
/// This test evaluates whether the distributions of two independent groups are
/// equal by ranking all observations and comparing the sum of ranks for each group.
///
/// # Arguments
///
/// * `data1` - An iterator over numeric values convertible to `f64` for the first group.
/// * `data2` - An iterator over numeric values convertible to `f64` for the second group.
/// * `alpha` - The significance level for the test, typically set at 0.05.
/// * `tail_type` - The type of tail for the test:
///   - `TailType::Left`: Test if the first group tends to have smaller values.
///   - `TailType::Right`: Test if the first group tends to have larger values.
///   - `TailType::Two`: Two-tailed test for difference in distributions.
///
/// # Returns
///
/// Returns a `Result<TestResult, String>`, where `TestResult` contains:
/// - `test_statistic`: The computed U statistic.
/// - `p_value`: The p-value for the test.
/// - `confidence_interval`: Not applicable for U test, returns `(NaN, NaN)`.
/// - `null_hypothesis`: The null hypothesis statement.
/// - `alt_hypothesis`: The alternative hypothesis statement.
/// - `reject_null`: Boolean indicating whether to reject the null hypothesis.
///
/// # Example
///
/// ```rust
/// use hypors::mann_whitney::u_test;
/// use hypors::common::TailType;
///
/// let group1 = vec![1.0, 2.0, 3.0, 4.0];
/// let group2 = vec![2.5, 3.5, 4.5];
/// let alpha = 0.05;
///
/// let result = u_test(group1.iter().copied(), group2.iter().copied(), alpha, TailType::Two).unwrap();
///
/// println!("U Statistic: {}", result.test_statistic);
/// println!("P-value: {}", result.p_value);
/// println!("Reject Null: {}", result.reject_null);
/// ```
pub fn u_test<I, J, T, U>(
    data1: I,
    data2: J,
    alpha: f64,
    tail_type: TailType,
) -> Result<TestResult, String>
where
    I: IntoIterator<Item = T>,
    J: IntoIterator<Item = U>,
    T: Into<f64>,
    U: Into<f64>,
{
    // Collect and convert data to f64 vectors
    let mut combined: Vec<(f64, u8)> = Vec::new();

    for val in data1.into_iter() {
        combined.push((val.into(), 1));
    }
    for val in data2.into_iter() {
        combined.push((val.into(), 2));
    }

    let n1 = combined.iter().filter(|(_, g)| *g == 1).count() as f64;
    let n2 = combined.iter().filter(|(_, g)| *g == 2).count() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return Err("Both groups must contain at least one observation.".to_string());
    }

    // Sort combined by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut rank_values = vec![0.0; combined.len()];
    let mut i = 0;

    // Assign ranks with tie handling (average rank)
    while i < combined.len() {
        let start = i;
        let mut end = i;

        while end + 1 < combined.len() && combined[end + 1].0 == combined[start].0 {
            end += 1;
        }

        let rank_avg = ((start + 1) + (end + 1)) as f64 / 2.0;
        for v in rank_values.iter_mut().take(end + 1).skip(start) {
            *v = rank_avg;
        }

        i = end + 1;
    }

    // Sum ranks for each group
    let mut rank_sum1 = 0.0;
    let mut rank_sum2 = 0.0;

    for (rank, group) in rank_values.iter().zip(combined.iter()) {
        if group.1 == 1 {
            rank_sum1 += rank;
        } else {
            rank_sum2 += rank;
        }
    }

    // Calculate U statistics for both groups
    let u1 = rank_sum1 - (n1 * (n1 + 1.0) / 2.0);
    let u2 = rank_sum2 - (n2 * (n2 + 1.0) / 2.0);
    let u_statistic = u1.min(u2);

    // Calculate p-value using normal approximation
    let total = n1 + n2;
    let mean_u = (n1 * n2) / 2.0;
    let variance_u = (n1 * n2 * (total + 1.0)) / 12.0;

    let z = (u_statistic - mean_u) / variance_u.sqrt();

    let dist = Normal::new(0.0, 1.0).map_err(|e| format!("Normal distribution error: {e}"))?;
    let p_value = calculate_p(z, tail_type, &dist);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic: u_statistic,
        p_value,
        confidence_interval: (f64::NAN, f64::NAN),
        null_hypothesis: "H0: The distributions of both groups are equal.".to_string(),
        alt_hypothesis: "Ha: The distributions of both groups are not equal.".to_string(),
        reject_null,
    })
}
