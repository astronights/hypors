use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use statrs::distribution::Normal;

/// Performs an independent two-sample Z-test for proportions.
///
/// This test evaluates whether the difference in proportions between two independent groups
/// is statistically significant.
///
/// # Arguments
///
/// * `data1` - Iterator of binary values for the first group (e.g., 0/1, bool).
/// * `data2` - Iterator of binary values for the second group.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05).
/// * `pooled` - Whether to use pooled proportions to calculate the standard error.
///
/// # Returns
///
/// A `TestResult` with the test statistic, p-value, confidence interval, null/alt hypotheses, and whether to reject null.
///
/// # Errors
///
/// Returns `StatError` if:
/// - Either sample is empty
/// - Standard error is zero
/// - Statistical computation fails
///
/// # Example
///
/// ```rust
/// use hypors::proportion::z_test_ind;
/// use hypors::common::TailType;
///
/// let group1 = vec![1, 0, 1, 1, 0];
/// let group2 = vec![0, 0, 1, 1, 1];
/// let result = z_test_ind(group1.iter().copied(), group2.iter().copied(), TailType::Two, 0.05, true).unwrap();
///
/// println!("Z Statistic: {}", result.test_statistic);
/// ```
pub fn z_test_ind<I1, I2, T>(
    data1: I1,
    data2: I2,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> Result<TestResult, StatError>
where
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
    T: Into<f64>,
{
    let sample1: Vec<f64> = data1.into_iter().map(|x| x.into()).collect();
    let sample2: Vec<f64> = data2.into_iter().map(|x| x.into()).collect();

    if sample1.is_empty() || sample2.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let successes1: f64 = sample1.iter().sum();
    let successes2: f64 = sample2.iter().sum();

    let p1 = successes1 / n1;
    let p2 = successes2 / n2;

    let std_error = if pooled {
        let pooled_p = (successes1 + successes2) / (n1 + n2);
        (pooled_p * (1.0 - pooled_p) * (1.0 / n1 + 1.0 / n2)).sqrt()
    } else {
        ((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2)).sqrt()
    };

    if std_error == 0.0 {
        return Err(StatError::ComputeError(
            "Standard error is zero; cannot compute test statistic".to_string(),
        ));
    }

    let test_statistic = (p1 - p2) / std_error;

    let z_dist = Normal::new(0.0, 1.0).map_err(|e| {
        StatError::ComputeError(format!("Failed to create Normal distribution: {e}"))
    })?;

    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(p1 - p2, std_error, alpha, &z_dist);
    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: p1 >= p2".to_string(),
        TailType::Right => "H0: p1 <= p2".to_string(),
        TailType::Two => "H0: p1 = p2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: p1 < p2".to_string(),
        TailType::Right => "Ha: p1 > p2".to_string(),
        TailType::Two => "Ha: p1 ≠ p2".to_string(),
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
