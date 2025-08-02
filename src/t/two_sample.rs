use crate::common::{StatError, TailType, TestResult, calculate_ci, calculate_p};
use crate::t::t_test;
use statrs::distribution::StudentsT;

/// Performs a paired two-sample t-test on two related samples.
///
/// This function evaluates whether the means of two related groups differ from each other.
/// It works by calculating the differences between paired observations and performing a one-sample t-test on those differences.
///
/// # Arguments
///
/// * `data1` - An iterator containing the first set of sample data.
/// * `data2` - An iterator containing the second set of sample data.
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
/// Returns a `StatError` if there are issues with the data (empty, insufficient, mismatched lengths) or calculations.
///
/// # Example
///
/// ```rust
/// use hypors::t::two_sample::t_test_paired;
/// use hypors::common::TailType;
///
/// let data1 = vec![1.2, 2.3, 1.9, 2.5, 2.8];
/// let data2 = vec![1.1, 2.0, 1.7, 2.3, 2.6];
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
///
/// // Perform the paired two-sample t-test
/// let result = t_test_paired(data1.iter().copied(), data2.iter().copied(), tail, alpha).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
///
/// # Example with arrays
///
/// ```rust
/// use hypors::t::t_test_paired;
/// use hypors::common::TailType;
///
/// let before = [120.0, 135.0, 140.0, 125.0, 130.0];
/// let after = [115.0, 130.0, 135.0, 120.0, 125.0];
///
/// let result = t_test_paired(before.iter().copied(), after.iter().copied(), TailType::Two, 0.05).unwrap();
/// ```
pub fn t_test_paired<I1, I2, T1, T2>(
    data1: I1,
    data2: I2,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError>
where
    I1: IntoIterator<Item = T1>,
    I2: IntoIterator<Item = T2>,
    T1: Into<f64>,
    T2: Into<f64>,
{
    // Convert iterators to Vec<f64>
    let sample1: Vec<f64> = data1.into_iter().map(|x| x.into()).collect();
    let sample2: Vec<f64> = data2.into_iter().map(|x| x.into()).collect();

    // Check that both samples have the same length
    if sample1.len() != sample2.len() {
        return Err(StatError::ComputeError(format!(
            "Sample sizes must be equal for paired t-test. Got {} and {}",
            sample1.len(),
            sample2.len()
        )));
    }

    // Calculate differences
    let differences: Vec<f64> = sample1
        .iter()
        .zip(sample2.iter())
        .map(|(x1, x2)| x1 - x2)
        .collect();

    // Perform one-sample t-test on differences against mean of 0
    let mut result = t_test(differences.iter().copied(), 0.0, tail.clone(), alpha)?;

    // Update hypothesis strings for paired test
    result.null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    result.alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 ≠ µ2".to_string(),
    };

    Ok(result)
}

/// Convenience function for paired t-test with `Vec<f64>` input
pub fn t_test_paired_vec(
    data1: &[f64],
    data2: &[f64],
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, StatError> {
    t_test_paired(data1.iter().copied(), data2.iter().copied(), tail, alpha)
}

/// Performs an independent two-sample t-test on two unrelated samples.
///
/// This function evaluates whether the means of two independent groups differ from each other.
/// It supports both pooled variance (standard t-test) and unequal variance (Welch's t-test) approaches.
///
/// # Arguments
///
/// * `data1` - An iterator containing the first set of sample data.
/// * `data2` - An iterator containing the second set of sample data.
/// * `tail` - The type of tail (left, right, or two) for the test.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `pooled` - Whether to pool variances (true for a standard t-test, false for Welch's t-test).
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
/// use hypors::t::t_test_ind;
/// use hypors::common::TailType;
///
/// let group1 = vec![1.2, 2.3, 1.9, 2.5, 2.8];
/// let group2 = vec![1.1, 2.0, 1.7, 2.3, 2.6];
/// let tail = TailType::Two; // Two-tailed test
/// let alpha = 0.05; // 5% significance level
/// let pooled = false; // Use Welch's t-test
///
/// // Perform the independent two-sample t-test
/// let result = t_test_ind(group1.iter().copied(), group2.iter().copied(), tail, alpha, pooled).unwrap();
///
/// // Check if the p-value is within valid range
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// // Verify if the null hypothesis should be rejected
/// assert_eq!(result.reject_null, result.p_value < alpha);
/// ```
///
/// # Example with different sample sizes
///
/// ```rust
/// use hypors::t::two_sample::t_test_ind;
/// use hypors::common::TailType;
///
/// let control = [12.0, 15.0, 14.0, 16.0, 13.0, 18.0];
/// let treatment = [20.0, 22.0, 19.0, 24.0]; // Different size is OK for independent samples
///
/// let result = t_test_ind(control.iter().copied(), treatment.iter().copied(),
///                        TailType::Two, 0.05, false).unwrap();
/// ```
pub fn t_test_ind<I1, I2, T1, T2>(
    data1: I1,
    data2: I2,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> Result<TestResult, StatError>
where
    I1: IntoIterator<Item = T1>,
    I2: IntoIterator<Item = T2>,
    T1: Into<f64>,
    T2: Into<f64>,
{
    // Convert iterators to Vec<f64>
    let sample1: Vec<f64> = data1.into_iter().map(|x| x.into()).collect();
    let sample2: Vec<f64> = data2.into_iter().map(|x| x.into()).collect();

    // Check for empty data
    if sample1.is_empty() || sample2.is_empty() {
        return Err(StatError::EmptyData);
    }

    // Check for insufficient data (need at least 2 points per group for variance)
    if sample1.len() < 2 || sample2.len() < 2 {
        return Err(StatError::InsufficientData);
    }

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Calculate means
    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    // Calculate variances (using n-1 denominator for unbiased estimate)
    let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    // Calculate standard error and degrees of freedom based on pooled vs unpooled
    let (std_error, df) = if pooled {
        // Pooled variance approach (standard t-test)
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let std_error = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
        let df = n1 + n2 - 2.0;
        (std_error, df)
    } else {
        // Welch's t-test (unequal variances)
        let std_error = (var1 / n1 + var2 / n2).sqrt();
        let df = (var1 / n1 + var2 / n2).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
        (std_error, df)
    };

    // Calculate test statistic
    let test_statistic = (mean1 - mean2) / std_error;

    // Create t-distribution
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
        StatError::ComputeError(format!("Failed to create StudentsT distribution: {e}"))
    })?;

    // Calculate p-value and confidence interval
    let p_value = calculate_p(test_statistic, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(mean1 - mean2, std_error, alpha, &t_dist);

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
