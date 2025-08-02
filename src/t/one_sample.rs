use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use statrs::distribution::StudentsT;

/// Performs a one-sample t-test on the provided data.
///
/// This function compares the mean of a sample to a known population mean to determine if there is a statistically significant difference.
///
/// # Arguments
///
/// * `data` - An iterator containing the sample data (any type that can be converted to f64).
/// * `pop_mean` - The population mean to test against.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// A `TestResult` struct containing the test statistic, p-value, confidence interval,
/// null/alternative hypotheses, and a boolean indicating whether the null hypothesis should be rejected.
///
/// # Errors
///
/// Returns a `StatError` if there are issues with the data (empty, insufficient) or calculations.
///
/// # Example
///
/// ```rust
/// use hypors::t::one_sample::t_test;
/// use hypors::common::TailType;
///
/// let data = vec![1.2, 2.3, 1.9, 2.5, 2.8];
/// let pop_mean = 2.0;
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
///
/// // Perform the one-sample t-test
/// let result = t_test(data.iter().copied(), pop_mean, tail, alpha).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
///
/// # Example with different data sources
///
/// ```rust
/// use hypors::t::t_test;
/// use hypors::common::TailType;
///
/// // Works with arrays
/// let array_data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let result1 = t_test(array_data.iter().copied(), 3.0, TailType::Two, 0.05).unwrap();
///
/// // Works with Vec
/// let vec_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result2 = t_test(vec_data.iter().copied(), 3.0, TailType::Two, 0.05).unwrap();
///
/// // Works with any iterator of numbers
/// let range_data = (1..=5).map(|x| x as f64);
/// let result3 = t_test(range_data, 3.0, TailType::Two, 0.05).unwrap();
/// ```
pub fn t_test<I, T>(
    data: I,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I: IntoIterator<Item = T>,
    T: Into<f64>,
{
    // Convert iterator to Vec<f64>
    let sample_data: Vec<f64> = data.into_iter().map(|x| x.into()).collect();

    // Check for empty data
    if sample_data.is_empty() {
        return Err(StatError::EmptyData);
    }

    // Check for insufficient data (need at least 2 points for variance)
    if sample_data.len() < 2 {
        return Err(StatError::InsufficientData);
    }

    let n = sample_data.len() as f64;

    // Calculate sample mean
    let sample_mean = sample_data.iter().sum::<f64>() / n;

    // Calculate sample variance (using n-1 denominator for unbiased estimate)
    let sample_var = sample_data
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let std_error = (sample_var / n).sqrt();

    // Calculate test statistic
    let test_statistic = (sample_mean - pop_mean) / std_error;
    let df = n - 1.0;

    // Create t-distribution
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        StatError::ComputeError(format!("Failed to create StudentsT distribution: {e}"))
    })?;

    // Calculate p-value and confidence interval
    let p_value = calculate_p(test_statistic, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &t_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ >= {pop_mean}"),
        TailType::Right => format!("H0: µ <= {pop_mean}"),
        TailType::Two => format!("H0: µ = {pop_mean}"),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: µ < {pop_mean}"),
        TailType::Right => format!("Ha: µ > {pop_mean}"),
        TailType::Two => format!("Ha: µ ≠ {pop_mean}"),
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
