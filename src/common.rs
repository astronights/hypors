use statrs::distribution::ContinuousCDF;
use serde::{Serialize, Deserialize};

pub enum TailType {
    Left,
    Right,
    Two,
}

fn calculate_p(t_stat: f64, tail: TailType, dist: ContinuousCDF) -> f64 {
    match tail {
        TailType::Left => dist.cdf(t_stat),
        TailType::Right => 1.0 - dist.cdf(t_stat),
        TailType::Two => 2.0 * (1.0 - dist.cdf(t_stat.abs())),
    }
}

fn calculate_ci(sample_mean: f64, std_error: f64, alpha: f64, dist: ContinuousCDF) -> (f64, f64) {
    let margin_of_error = dist.inverse_cdf(1.0 - alpha / 2.0) * std_error;
    (sample_mean - margin_of_error, sample_mean + margin_of_error)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub null_hypothesis: String,
    pub alt_hypothesis: String,
    pub reject_null: bool,
}