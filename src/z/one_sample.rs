use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use statrs::distribution::Normal;

/// Performs a one-sample Z-test on the provided data.
///
/// The one-sample Z-test evaluates whether the mean of a sample differs significantly
/// from a known population mean, when the population standard deviation is known.
/// This test assumes the sampling distribution of the mean is approximately normal,
/// which is valid for large sample sizes (n ≥ 30) or when the population is normally distributed.
///
/// # Arguments
///
/// * `data` - An iterator containing the sample data (any type that can be converted to f64).
/// * `pop_mean` - The known population mean to test against.
/// * `pop_std` - The known population standard deviation (must be positive).
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
/// Returns a `StatError` if:
/// - The data is empty (`EmptyData`)
/// - The population standard deviation is not positive (`ComputeError`)
/// - There are issues with statistical calculations (`ComputeError`)
///
/// # Statistical Background
///
/// The Z-test statistic is calculated as:
/// ```text
/// Z = (sample_mean - population_mean) / (population_std / √n)
/// ```
///
/// Where:
/// - `sample_mean` is the mean of the sample data
/// - `population_mean` is the hypothesized population mean
/// - `population_std` is the known population standard deviation
/// - `n` is the sample size
///
/// # Example
///
/// ```rust
/// use hypors::z::z_test;
/// use hypors::common::TailType;
///
/// let data = vec![1.2, 2.3, 1.9, 2.5, 2.8, 2.1, 1.8, 2.4, 2.0, 2.6];
/// let pop_mean = 2.0;           // Known population mean
/// let pop_std = 0.5;            // Known population standard deviation
/// let tail = TailType::Two;     // Two-tailed test
/// let alpha = 0.05;             // 5% significance level
///
/// // Perform the one-sample Z-test
/// let result = z_test(data.iter().copied(), pop_mean, pop_std, tail, alpha).unwrap();
///
/// // Check if the p-value is within a valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
///
/// # Example with different data sources
///
/// ```rust
/// use hypors::z::one_sample::z_test;
/// use hypors::common::TailType;
///
/// // Works with arrays
/// let array_data = [98.6, 99.1, 98.8, 99.3, 98.9, 99.0, 98.7, 99.2];
/// let result1 = z_test(array_data.iter().copied(), 99.0, 0.3, TailType::Two, 0.05).unwrap();
///
/// // Works with Vec
/// let vec_data = vec![150.0, 155.0, 148.0, 152.0, 149.0, 153.0];
/// let result2 = z_test(vec_data.iter().copied(), 150.0, 5.0, TailType::Right, 0.05).unwrap();
///
/// // Works with any iterator of numbers
/// let range_data = (95..=105).map(|x| x as f64);
/// let result3 = z_test(range_data, 100.0, 3.0, TailType::Two, 0.01).unwrap();
/// ```
///
/// # When to use Z-test vs t-test
///
/// - **Use Z-test when**: Population standard deviation is known, large sample size (n ≥ 30)
/// - **Use t-test when**: Population standard deviation is unknown, small sample size
pub fn z_test<I, T>(
    data: I,
    pop_mean: f64,
    pop_std: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I: IntoIterator<Item = T>,
    T: Into<f64>,
{
    // Validate population standard deviation
    if pop_std <= 0.0 {
        return Err(StatError::ComputeError(format!(
            "Population standard deviation must be positive, got: {pop_std}",
        )));
    }

    // Convert iterator to Vec<f64>
    let sample_data: Vec<f64> = data.into_iter().map(|x| x.into()).collect();

    // Check for empty data
    if sample_data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n = sample_data.len() as f64;

    // Calculate sample mean
    let sample_mean = sample_data.iter().sum::<f64>() / n;

    // Calculate standard error of the mean
    let std_error = pop_std / n.sqrt();

    // Calculate Z test statistic
    let test_statistic = (sample_mean - pop_mean) / std_error;

    // Create standard normal distribution
    let z_dist = Normal::new(0.0, 1.0).map_err(|e| {
        StatError::ComputeError(format!("Failed to create Normal distribution: {e}"))
    })?;

    // Calculate p-value and confidence interval
    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &z_dist);

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
