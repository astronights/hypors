use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use statrs::distribution::Normal;

/// Performs a one-sample proportion Z-test on the provided binary data.
///
/// # Arguments
///
/// * `data` - An iterator over binary values (0 or 1), where 1 represents success.
/// * `pop_proportion` - The hypothesized population proportion (between 0 and 1).
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// A `TestResult` containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and whether to reject the null hypothesis.
///
/// # Errors
///
/// Returns `StatError` if:
/// - The data is empty
/// - The population proportion is not between 0 and 1
/// - Statistical computation fails
///
/// # Example
///
/// ```rust
/// use hypors::proportion::z_test;
/// use hypors::common::TailType;
///
/// let data = vec![1, 0, 1, 1, 0, 1, 0, 0];
/// let result = z_test(data.iter().copied(), 0.5, TailType::Two, 0.05).unwrap();
///
/// println!("Z Statistic: {}", result.test_statistic);
/// ```
pub fn z_test<I, T>(
    data: I,
    pop_proportion: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I: IntoIterator<Item = T>,
    T: Into<f64>,
{
    if !(0.0..=1.0).contains(&pop_proportion) {
        return Err(StatError::ComputeError(format!(
            "Population proportion must be between 0 and 1, got: {pop_proportion}"
        )));
    }

    let sample: Vec<f64> = data.into_iter().map(|x| x.into()).collect();

    if sample.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n = sample.len() as f64;
    let successes: f64 = sample.iter().sum();
    let sample_proportion = successes / n;

    let std_error = (pop_proportion * (1.0 - pop_proportion) / n).sqrt();

    if std_error == 0.0 {
        return Err(StatError::ComputeError(
            "Standard error is zero; cannot compute test statistic".to_string(),
        ));
    }

    let test_statistic = (sample_proportion - pop_proportion) / std_error;

    let z_dist = Normal::new(0.0, 1.0).map_err(|e| {
        StatError::ComputeError(format!("Failed to create Normal distribution: {e}"))
    })?;

    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(sample_proportion, std_error, alpha, &z_dist);
    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: p >= {pop_proportion}"),
        TailType::Right => format!("H0: p <= {pop_proportion}"),
        TailType::Two => format!("H0: p = {pop_proportion}"),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: p < {pop_proportion}"),
        TailType::Right => format!("Ha: p > {pop_proportion}"),
        TailType::Two => format!("Ha: p ≠ {pop_proportion}"),
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
