use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Calculates the required sample size for a chi-square goodness-of-fit test.
///
/// This function computes the necessary sample size to ensure each expected frequency is adequate
/// for the goodness-of-fit test.
///
/// # Arguments
///
/// * `expected_counts` - A vector of expected counts for each category.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::{chi2_sample_size_gof};
///
/// let expected_counts = vec![20, 30, 50];
/// let alpha = 0.05; // 5% significance level
///
/// let sample_size = chi2_sample_size_gof(&expected_counts, alpha, power);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn chi2_sample_size_gof(expected_counts: &[usize], alpha: f64) -> f64 {
    let total_expected: f64 = expected_counts.iter().map(|&count| count as f64).sum();
    let expected_proportions: Vec<f64> = expected_counts
        .iter()
        .map(|&count| count as f64 / total_expected)
        .collect();

    let chi_squared_dist = ChiSquared::new(expected_counts.len() as f64 - 1.0)
        .expect("Failed to create Chi-squared distribution");

    let chi_alpha = chi_squared_dist.inverse_cdf(1.0 - alpha);

    // Using the formula n = (chi_alpha / expected_proportion)^2
    let n = expected_proportions
        .iter()
        .map(|&prop| (chi_alpha / prop).powi(2))
        .fold(0.0, |sum, x| sum + x)
        / expected_counts.len() as f64;

    n.ceil() // Rounds up to the next whole sample size
}

/// Calculates the required sample size for a chi-square test for independence.
///
/// This function computes the necessary sample size to detect a minimum detectable effect size
/// for a given alpha, power, and expected proportions in a contingency table.
///
/// # Arguments
///
/// * `expected_counts` - A vector of expected counts for each cell in the contingency table.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::{chi2_sample_size_ind};
///
/// let expected_counts = vec![20, 30, 50];
/// let alpha = 0.05; // 5% significance level
///
/// let sample_size = chi2_sample_size_ind(&expected_counts, alpha, power);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn chi2_sample_size_ind(expected_counts: &[usize], alpha: f64) -> f64 {
    let total_expected: f64 = expected_counts.iter().map(|&count| count as f64).sum();
    let expected_proportions: Vec<f64> = expected_counts
        .iter()
        .map(|&count| count as f64 / total_expected)
        .collect();

    let chi_squared_dist = ChiSquared::new(expected_counts.len() as f64 - 1.0)
        .expect("Failed to create Chi-squared distribution");

    let chi_alpha = chi_squared_dist.inverse_cdf(1.0 - alpha);

    // Using the formula n = (chi_alpha / expected_proportion)^2
    let n = expected_proportions
        .iter()
        .map(|&prop| (chi_alpha / prop).powi(2))
        .fold(0.0, |sum, x| sum + x)
        / expected_counts.len() as f64;

    n.ceil() // Rounds up to the next whole sample size
}

/// Calculates the required sample size for a chi-square test for variance.
///
/// This function computes the necessary sample size to detect a minimum detectable effect size
/// for a given alpha, power, and population variance.
///
/// # Arguments
///
/// * `effect_size` - The minimum detectable effect size (in terms of variance).
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `power` - The desired statistical power (e.g., 0.80 for 80% power).
/// * `variance` - The population variance (or a reasonable estimate).
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::{chi2_sample_size_variance};
///
/// let effect_size = 0.5; // Minimum detectable effect size
/// let alpha = 0.05; // 5% significance level
/// let power = 0.80; // 80% power
/// let variance = 1.0; // Population variance
///
/// let sample_size = chi2_sample_size_variance(effect_size, alpha, power, variance);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn chi2_sample_size_variance(effect_size: f64, alpha: f64, power: f64, variance: f64) -> f64 {
    let chi_squared_dist = ChiSquared::new(1.0).expect("Failed to create Chi-squared distribution");

    let chi_alpha = chi_squared_dist.inverse_cdf(1.0 - alpha);
    let chi_beta = chi_squared_dist.inverse_cdf(power);

    // Formula: n = ((chi_alpha + chi_beta) * variance / effect_size)^2
    let n = ((chi_alpha + chi_beta) * variance / effect_size).powi(2);
    n.ceil() // Rounds up to the next whole sample size
}
