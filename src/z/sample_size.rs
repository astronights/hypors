use crate::common::TailType;
use statrs::distribution::{ContinuousCDF, Normal};

/// Calculates the required sample size for a one-sample Z-test.
///
/// This function computes the necessary sample size to detect a minimum detectable effect size
/// for a given alpha, power, and standard deviation.
///
/// # Arguments
///
/// * `effect_size` - The minimum detectable effect size.
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `power` - The desired statistical power (e.g., 0.80 for 80% power).
/// * `std_dev` - The population standard deviation (or a reasonable estimate).
/// * `tail` - The type of tail (left, right, or two) for the test.
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::z::z_sample_size;
/// use hypors::common::TailType;
///
/// let effect_size = 0.5;
/// let alpha = 0.05; // 5% significance level
/// let power = 0.80; // 80% power
/// let std_dev = 1.0;
/// let tail = TailType::Two; // Two-tailed test
///
/// let sample_size = z_sample_size(effect_size, alpha, power, std_dev, tail);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn z_sample_size(
    effect_size: f64,
    alpha: f64,
    power: f64,
    std_dev: f64,
    tail: TailType,
) -> f64 {
    // Z-distribution for critical values
    let normal_dist = Normal::new(0.0, 1.0).expect("Failed to create Normal distribution");

    let alpha_value = match tail {
        TailType::Two => alpha / 2.0, // Two-tailed
        _ => alpha,                   // One-tailed (left or right)
    };

    let z_alpha = normal_dist.inverse_cdf(1.0 - alpha_value);
    let z_beta = normal_dist.inverse_cdf(power);

    // Formula: n = ((z_alpha + z_beta) * std_dev / effect_size)^2
    let n = ((z_alpha + z_beta) * std_dev / effect_size).powi(2);
    n.ceil() // Rounds up to the next whole sample size
}
