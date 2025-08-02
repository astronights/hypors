use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use statrs::distribution::Normal;

/// Performs a paired two-sample Z-test on two related samples.
///
/// The paired Z-test evaluates whether the mean difference between two related samples
/// differs significantly from zero, when the population standard deviation of the differences
/// is known. This test is appropriate when observations come in pairs (e.g., before/after
/// measurements on the same subjects).
///
/// # Arguments
///
/// * `data1` - An iterator containing the first set of sample data.
/// * `data2` - An iterator containing the second set of sample data.
/// * `pop_std_diff` - The population standard deviation of the differences between the samples (must be positive).
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
/// - Either dataset is empty (`EmptyData`)
/// - The datasets have different lengths (`ComputeError`)
/// - The population standard deviation is not positive (`ComputeError`)
/// - There are issues with statistical calculations (`ComputeError`)
///
/// # Statistical Background
///
/// The paired Z-test statistic is calculated as:
/// ```text
/// Z = (mean_difference - 0) / (pop_std_diff / √n)
/// ```
///
/// Where:
/// - `mean_difference` is the mean of the paired differences
/// - `pop_std_diff` is the known population standard deviation of differences
/// - `n` is the number of pairs
///
/// # Example
///
/// ```rust
/// use hypors::z::z_test_paired;
/// use hypors::common::TailType;
///
/// let before = vec![120.0, 118.0, 125.0, 122.0, 130.0, 128.0];
/// let after = vec![115.0, 112.0, 120.0, 118.0, 125.0, 123.0];
/// let pop_std_diff = 3.0;       // Known population std of differences
/// let tail = TailType::Two;     // Two-tailed test
/// let alpha = 0.05;             // 5% significance level
///
/// let result = z_test_paired(
///     before.iter().copied(),
///     after.iter().copied(),
///     pop_std_diff,
///     tail,
///     alpha
/// ).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn z_test_paired<I1, I2, T1, T2>(
    data1: I1,
    data2: I2,
    pop_std_diff: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I1: IntoIterator<Item = T1>,
    I2: IntoIterator<Item = T2>,
    T1: Into<f64>,
    T2: Into<f64>,
{
    // Validate population standard deviation
    if pop_std_diff <= 0.0 {
        return Err(StatError::ComputeError(format!(
            "Population standard deviation must be positive, got: {pop_std_diff}"
        )));
    }

    // Convert iterators to Vec<f64>
    let sample1: Vec<f64> = data1.into_iter().map(|x| x.into()).collect();
    let sample2: Vec<f64> = data2.into_iter().map(|x| x.into()).collect();

    // Check for empty data
    if sample1.is_empty() || sample2.is_empty() {
        return Err(StatError::EmptyData);
    }

    // Check for equal lengths
    if sample1.len() != sample2.len() {
        return Err(StatError::ComputeError(format!(
            "Sample sizes must be equal: {} vs {}",
            sample1.len(),
            sample2.len()
        )));
    }

    let n = sample1.len() as f64;

    // Calculate differences
    let differences: Vec<f64> = sample1
        .iter()
        .zip(sample2.iter())
        .map(|(x1, x2)| x1 - x2)
        .collect();

    // Calculate mean of differences
    let sample_mean_diff = differences.iter().sum::<f64>() / n;

    // Calculate standard error
    let std_error = pop_std_diff / n.sqrt();

    // Calculate Z test statistic
    let test_statistic = sample_mean_diff / std_error;

    // Create standard normal distribution
    let z_dist = Normal::new(0.0, 1.0).map_err(|e| {
        StatError::ComputeError(format!("Failed to create Normal distribution: {e}"))
    })?;

    // Calculate p-value and confidence interval
    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(sample_mean_diff, std_error, alpha, &z_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
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

/// Performs an independent two-sample Z-test on two unrelated samples.
///
/// The independent two-sample Z-test evaluates whether the means of two independent samples
/// differ significantly, when the population standard deviations are known. This test assumes
/// the samples are independent and the sampling distributions are approximately normal.
///
/// # Arguments
///
/// * `data1` - An iterator containing the first set of sample data.
/// * `data2` - An iterator containing the second set of sample data.
/// * `pop_std1` - The population standard deviation for the first sample (must be positive).
/// * `pop_std2` - The population standard deviation for the second sample (must be positive).
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
/// - Either dataset is empty (`EmptyData`)
/// - Either population standard deviation is not positive (`ComputeError`)
/// - There are issues with statistical calculations (`ComputeError`)
///
/// # Statistical Background
///
/// The independent two-sample Z-test statistic is calculated as:
/// ```text
/// Z = (mean1 - mean2) / √((σ1²/n1) + (σ2²/n2))
/// ```
///
/// Where:
/// - `mean1`, `mean2` are the sample means
/// - `σ1`, `σ2` are the known population standard deviations
/// - `n1`, `n2` are the sample sizes
///
/// # Example
///
/// ```rust
/// use hypors::z::z_test_ind;
/// use hypors::common::TailType;
///
/// let group1 = vec![85.0, 88.0, 92.0, 87.0, 90.0, 89.0, 91.0];
/// let group2 = vec![78.0, 82.0, 80.0, 85.0, 79.0, 83.0];
/// let pop_std1 = 4.0;           // Known population std for group 1
/// let pop_std2 = 3.5;           // Known population std for group 2
/// let tail = TailType::Two;     // Two-tailed test
/// let alpha = 0.05;             // 5% significance level
///
/// let result = z_test_ind(
///     group1.iter().copied(),
///     group2.iter().copied(),
///     pop_std1,
///     pop_std2,
///     tail,
///     alpha
/// ).unwrap();
///
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
pub fn z_test_ind<I1, I2, T1, T2>(
    data1: I1,
    data2: I2,
    pop_std1: f64,
    pop_std2: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I1: IntoIterator<Item = T1>,
    I2: IntoIterator<Item = T2>,
    T1: Into<f64>,
    T2: Into<f64>,
{
    // Validate population standard deviations
    if pop_std1 <= 0.0 {
        return Err(StatError::ComputeError(format!(
            "Population standard deviation 1 must be positive, got: {pop_std1}",
        )));
    }
    if pop_std2 <= 0.0 {
        return Err(StatError::ComputeError(format!(
            "Population standard deviation 2 must be positive, got: {pop_std2}"
        )));
    }

    // Convert iterators to Vec<f64>
    let sample1: Vec<f64> = data1.into_iter().map(|x| x.into()).collect();
    let sample2: Vec<f64> = data2.into_iter().map(|x| x.into()).collect();

    // Check for empty data
    if sample1.is_empty() {
        return Err(StatError::EmptyData);
    }
    if sample2.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Calculate sample means
    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    // Calculate standard error
    let std_error = ((pop_std1.powi(2) / n1) + (pop_std2.powi(2) / n2)).sqrt();

    // Calculate Z test statistic
    let test_statistic = (mean1 - mean2) / std_error;

    // Create standard normal distribution
    let z_dist = Normal::new(0.0, 1.0).map_err(|e| {
        StatError::ComputeError(format!("Failed to create Normal distribution: {e}"))
    })?;

    // Calculate p-value and confidence interval
    let p_value = calculate_p(test_statistic, tail.clone(), &z_dist);
    let confidence_interval = calculate_ci(mean1 - mean2, std_error, alpha, &z_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    let alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
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
