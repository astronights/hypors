use crate::common::TailType;
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub fn calculate_p_value(t_stat: f64, tail: TailType, dist: &dyn ContinuousCDF<f64, f64>) -> f64 {
    match tail {
        TailType::Left => dist.cdf(t_stat),
        TailType::Right => 1.0 - dist.cdf(t_stat),
        TailType::Two => 2.0 * (1.0 - dist.cdf(t_stat.abs())),
    }
}

pub fn calculate_confidence_interval(
    sample_mean: f64,
    std_error: f64,
    alpha: f64,
    dist: &dyn ContinuousCDF<f64, f64>,
) -> (f64, f64) {
    let margin_of_error = dist.inverse_cdf(1.0 - alpha / 2.0) * std_error;
    (sample_mean - margin_of_error, sample_mean + margin_of_error)
}

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
