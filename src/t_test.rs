use crate::common::{calculate_ci,calculate_p,TailType,TestResult};
use polars::prelude::*;
use statrs::distribution::StudentsT;

pub fn one_sample(
    series: &Series,
    pop_mean: f64,
    tail: TailType,
    alpha: f64,
) -> Result<TestResult, PolarsError> {
    let sample_mean = series.mean().ok_or(PolarsError::ComputeError)?;  // Error if NaN
    let sample_var = series.var().ok_or(PolarsError::ComputeError)?;    // Error if NaN
    let n = series.len() as f64;
    let std_error = (sample_var / n).sqrt();

    let t_stat = (sample_mean - pop_mean) / std_error;
    let df = n - 1.0;
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    let p_value = calculate_p(t_stat, tail, t_dist.clone());
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
        t_stat,
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
    let diff_series = series1 - series2;

    let mut result = one_sample(&diff_series, 0.0, tail, alpha)?;

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
) -> Result<TTestResult, PolarsError> {
    let mean1 = series1.mean().ok_or(PolarsError::ComputeError)?;  // Error if NaN
    let mean2 = series2.mean().ok_or(PolarsError::ComputeError)?;
    let var1 = series1.var().ok_or(PolarsError::ComputeError)?;
    let var2 = series2.var().ok_or(PolarsError::ComputeError)?;
    let n1 = series1.len() as f64;
    let n2 = series2.len() as f64;

    let (std_error, df) = if pooled {
        let pooled_var = (((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0);
        let std_error = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
        let df = n1 + n2 - 2.0;
        (std_error, df)
    } else {
        let std_error = ((var1 / n1) + (var2 / n2)).sqrt();
        let df = ((var1 / n1) + (var2 / n2)).powi(2)
                 / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
        (std_error, df)
    };

    let t_stat = (mean1 - mean2) / std_error;
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    let p_value = calculate_p(t_stat, tail, t_dist.clone());

    let confidence_interval = calculate_ci(mean1 - mean2, std_error, alpha, &t_dist);

    let reject_null = p_value < alpha;

    let null_hypothesis = match tail {
        TailType::Left => format!("H0: µ1 >= µ2"),
        TailType::Right => format!("H0: µ1 <= µ2"),
        TailType::Two => format!("H0: µ1 = µ2"),
    };

    let alt_hypothesis = match tail {
        TailType::Left => format!("Ha: µ1 < µ2"),
        TailType::Right => format!("Ha: µ1 > µ2"),
        TailType::Two => format!("Ha: µ1 != µ2"),
    };

    Ok(TestResult {
        t_stat,
        p_value,
        confidence_interval,
        null_hypothesis,
        alt_hypothesis,
        reject_null,
    })
}
