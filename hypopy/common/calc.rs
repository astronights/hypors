use crate::common::TailType;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Calculates the p-value for a given test statistic.
///
/// This function determines the p-value based on the provided test statistic,
/// the type of tail (left, right, or two), and the statistical distribution used.
///
/// # Arguments
///
/// * `t_stat` - The test statistic (e.g., t-statistic).
/// * `tail` - The type of tail (left, right, or two).
/// * `dist` - The statistical distribution to be used, which must implement the `ContinuousCDF` trait.
///
/// # Returns
///
/// The p-value corresponding to the test statistic and tail type.
///
/// # Example
///
/// ```rust
/// use statrs::distribution::{StudentsT, ContinuousCDF};
/// use hypors::TailType;
/// use hypors::calculate_p_value;
///
/// let t_stat = 2.0;
/// let tail = TailType::Two;
/// let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();  // Student's t-distribution with 10 degrees of freedom
///
/// let p_value = calculate_p_value(t_stat, tail, &t_dist);
/// assert!(p_value > 0.0 && p_value < 1.0);
/// ```
pub fn calculate_p_value(t_stat: f64, tail: TailType, dist: &dyn ContinuousCDF<f64, f64>) -> f64 {
    match tail {
        TailType::Left => dist.cdf(t_stat),
        TailType::Right => 1.0 - dist.cdf(t_stat),
        TailType::Two => 2.0 * (1.0 - dist.cdf(t_stat.abs())),
    }
}

/// Calculates the confidence interval for a sample mean.
///
/// This function computes the confidence interval for a sample mean based on
/// the provided sample mean, standard error, significance level, and statistical distribution.
///
/// # Arguments
///
/// * `sample_mean` - The sample mean for the dataset.
/// * `std_error` - The standard error of the mean.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `dist` - The statistical distribution to be used, which must implement the `ContinuousCDF` trait.
///
/// # Returns
///
/// A tuple `(lower_bound, upper_bound)` representing the confidence interval.
///
/// # Example
///
/// ```rust
/// use statrs::distribution::{StudentsT, ContinuousCDF};
/// use hypors::calculate_confidence_interval;
///
/// let sample_mean = 5.0;
/// let std_error = 1.5;
/// let alpha = 0.05;
/// let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();  // Student's t-distribution with 10 degrees of freedom
///
/// let ci = calculate_confidence_interval(sample_mean, std_error, alpha, &t_dist);
/// assert!(ci.0 < sample_mean && ci.1 > sample_mean);  // Lower and upper bounds should surround the mean
/// ```
pub fn calculate_confidence_interval(
    sample_mean: f64,
    std_error: f64,
    alpha: f64,
    dist: &dyn ContinuousCDF<f64, f64>,
) -> (f64, f64) {
    let margin_of_error = dist.inverse_cdf(1.0 - alpha / 2.0) * std_error;
    (sample_mean - margin_of_error, sample_mean + margin_of_error)
}

/// Calculates the confidence interval for Chi-squared distribution.
///
/// This function computes the confidence interval for the variance of a population
/// based on the sample variance and the Chi-squared distribution.
///
/// # Arguments
///
/// * `sample_variance` - The sample variance for the dataset.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `dist` - The Chi-squared distribution used for the calculation.
///
/// # Returns
///
/// A tuple `(lower_bound, upper_bound)` representing the confidence interval for variance.
///
/// # Example
///
/// ```rust
/// use statrs::distribution::ChiSquared;
/// use hypors::calculate_chi2_confidence_interval;
///
/// let sample_variance = 2.5;
/// let alpha = 0.05;
/// let chi_squared_dist = ChiSquared::new(10.0).unwrap();  // Chi-squared distribution with 10 degrees of freedom
///
/// let ci = calculate_chi2_confidence_interval(sample_variance, alpha, &chi_squared_dist);
/// assert!(ci.0 < sample_variance && ci.1 > sample_variance); // Lower and upper bounds should surround the variance
/// ```
pub fn calculate_chi2_confidence_interval(
    sample_variance: f64,
    alpha: f64,
    dist: &ChiSquared,
) -> (f64, f64) {
    let df = dist.shape(); // Degrees of freedom
    let chi_square_lower = dist.inverse_cdf(alpha / 2.0);
    let chi_square_upper = dist.inverse_cdf(1.0 - alpha / 2.0);

    // Confidence interval for variance: (n-1) * sample_variance / chi_square_stat
    let lower_bound = (df * sample_variance) / chi_square_upper;
    let upper_bound = (df * sample_variance) / chi_square_lower;
    (lower_bound, upper_bound)
}
