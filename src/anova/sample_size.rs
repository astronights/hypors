use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Calculates the required sample size for an ANOVA test.
///
/// This function computes the necessary sample size to detect a minimum detectable effect size
/// for a given alpha, power, and number of groups.
///
/// # Arguments
///
/// * `effect_size` - The minimum detectable effect size (Cohen's f).
/// * `alpha` - The significance level (e.g., 0.05 for a 95% confidence interval).
/// * `power` - The desired statistical power (e.g., 0.80 for 80% power).
/// * `num_groups` - The number of groups in the ANOVA.
///
/// # Returns
///
/// The estimated sample size required to achieve the specified power and significance level.
///
/// # Example
/// ```rust
/// use hypors::{f_sample_size};
///
/// let effect_size = 0.25; // Cohen's f
/// let alpha = 0.05; // 5% significance level
/// let power = 0.80; // 80% power
/// let num_groups = 3; // Number of groups
///
/// let sample_size = f_sample_size(effect_size, alpha, power, num_groups);
/// println!("Required sample size: {}", sample_size);
/// ```
pub fn f_sample_size(effect_size: f64, alpha: f64, power: f64, num_groups: usize) -> f64 {
    let df1 = (num_groups - 1) as f64; // Degrees of freedom for the numerator
    let df2 = 1e6; // Approximation for large sample sizes (will be refined later)

    let f_dist = FisherSnedecor::new(df1, df2).expect("Failed to create Fisher's F distribution");

    // Calculate critical F-values based on alpha and power
    let f_alpha = f_dist.inverse_cdf(1.0 - alpha);
    let f_beta = f_dist.inverse_cdf(power);

    // Formula: n = ((f_alpha + f_beta) * (num_groups - 1) / effect_size^2)^2
    let n = ((f_alpha + f_beta) * df1 / effect_size.powi(2)).powi(2);
    n.ceil() // Rounds up to the next whole sample size
}
