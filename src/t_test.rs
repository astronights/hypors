use crate::common::{calculate_ci, calculate_p, TailType, TestResult};
use polars::prelude::PolarsError;
use polars::prelude::*;
use statrs::distribution::StudentsT; // Import the necessary Polars errors.

pub fn one_sample(
    series: &Series,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let sample_mean = series
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample mean".into()))?; // Error if NaN
    let sample_var = series
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute sample variance".into()))?; // Error if NaN

    let n = series.len() as f64;
    let std_error = (sample_var / n).sqrt();

    let t_stat = (sample_mean - pop_mean) / std_error;
    let df = n - 1.0;

    // Handle StudentsT unwrap properly
    let t_dist = StudentsT::new(0.0, 1.0, df).expect("Failed to create StudentsT distribution");

    // Use the correct reference when calling calculate_p and calculate_ci
    let p_value = calculate_p(t_stat, tail.clone(), &t_dist);
    let confidence_interval = calculate_ci(sample_mean, std_error, alpha, &t_dist);

    let reject_null = p_value < alpha;

    // Hypotheses strings
    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ >= {}", pop_mean),
        TailType::Right => format!("H0: µ <= {}", pop_mean),
        TailType::Two => format!("H0: µ = {}", pop_mean),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: µ < {}", pop_mean),
        TailType::Right => format!("Ha: µ > {}", pop_mean),
        TailType::Two => format!("Ha: µ != {}", pop_mean),
    };

    Ok(TestResult {
        test_statistic: t_stat,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}

pub fn two_sample_paired(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let diff_series = (series1 - series2).expect("Unable to get Series difference");

    let mut result = one_sample(&diff_series, 0.0, tail.clone(), alpha)?;

    result.null_hypothesis = match tail {
        TailType::Left => "H0: µ1 >= µ2".to_string(),
        TailType::Right => "H0: µ1 <= µ2".to_string(),
        TailType::Two => "H0: µ1 = µ2".to_string(),
    };

    result.alt_hypothesis = match tail {
        TailType::Left => "Ha: µ1 < µ2".to_string(),
        TailType::Right => "Ha: µ1 > µ2".to_string(),
        TailType::Two => "Ha: µ1 != µ2".to_string(),
    };

    Ok(result)
}

pub fn two_sample_ind(
    series1: &Series,
    series2: &Series,
    tail: TailType,
    alpha: f64,
    pooled: bool,
) -> Result<TestResult, PolarsError> {
    let mean1 = series1
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;
    let mean2 = series2
        .mean()
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute mean".into()))?;
    let var1 = series1
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute variance".into()))?;
    let var2 = series2
        .var(1)
        .ok_or_else(|| PolarsError::ComputeError("Failed to compute variance".into()))?;

    let n1 = series1.len() as f64;
    let n2 = series2.len() as f64;

    let (std_error, df) = if pooled {
        // Pooled variance
        let pooled_var = (((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0);
        let std_error = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
        let df = n1 + n2 - 2.0;
        (std_error, df)
    } else {
        // Unpooled (Welch's t-test)
        let std_error = ((var1 / n1) + (var2 / n2)).sqrt();
        let df = ((var1 / n1) + (var2 / n2)).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
        (std_error, df)
    };

    let test_statistic = (mean1 - mean2) / std_error;
    let t_dist = StudentsT::new(0.0, 1.0, df).expect("Failed to create StudentsT distribution");

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
        TailType::Two => "Ha: µ1 != µ2".to_string(),
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
