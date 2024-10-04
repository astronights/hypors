use statrs::distribution::{ContinuousCDF, Normal};

/// Calculates the required sample size for a test of proportions.
///
/// This function computes the necessary sample size to detect a minimum detectable difference
/// in proportions for a given alpha, power, and expected proportions.
///
/// # Arguments
///
/// * `p1` - The expected proportion in the first group.
/// * `p2` - The expected proportion in the second group.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `power` - The desired statistical power (e.g., 0.80 for 80% power).
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::{prop_sample_size};
///
/// let p1 = 0.4; // Expected proportion in group 1
/// let p2 = 0.5; // Expected proportion in group 2
/// let alpha = 0.05; // 5% significance level
/// let power = 0.80; // 80% power
///
/// let sample_size = prop_sample_size(p1, p2, alpha, power);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn prop_sample_size(p1: f64, p2: f64, alpha: f64, power: f64) -> f64 {
    let p = (p1 + p2) / 2.0; // Pooled proportion
    let z_alpha = Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(1.0 - alpha / 2.0); // Two-tailed
    let z_beta = Normal::new(0.0, 1.0).unwrap().inverse_cdf(power);

    // Formula: n = ((z_alpha * sqrt(2 * p * (1 - p)) + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2)))^2) / (p2 - p1)^2
    let n = (z_alpha * (2.0 * p * (1.0 - p)).sqrt()
        + z_beta * ((p1 * (1.0 - p1)) + (p2 * (1.0 - p2))).sqrt())
    .powi(2)
        / (p2 - p1).powi(2);

    n.ceil() // Rounds up to the next whole sample size
}
