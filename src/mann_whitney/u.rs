use crate::common::{calculate_p, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::Normal;
use std::f64;

pub fn u_test(
    data1: &Series,
    data2: &Series,
    alpha: f64,
    tail_type: TailType,
) -> Result<TestResult, PolarsError> {
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;

    // Combine the data
    let mut combined = Vec::new();
    for value in data1.f64()?.into_iter() {
        if let Some(v) = value {
            combined.push(v);
        }
    }
    for value in data2.f64()?.into_iter() {
        if let Some(v) = value {
            combined.push(v);
        }
    }

    // Rank the data
    let mut ranks: Vec<(f64, usize)> = combined.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    ranks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut rank_sum1 = 0.0;
    let mut rank_sum2 = 0.0;

    for (rank, (_value, original_index)) in ranks.iter().enumerate() {
        if original_index < &(n1 as usize) {
            rank_sum1 += (rank as f64) + 1.0; // Ranks start at 1
        } else {
            rank_sum2 += (rank as f64) + 1.0;
        }
    }

    // Calculate U statistic
    let u1 = rank_sum1 - (n1 * (n1 + 1.0) / 2.0);
    let u2 = rank_sum2 - (n2 * (n2 + 1.0) / 2.0);
    let u_statistic = u1.min(u2);

    // Calculate p-value (based on tail type)
    let total = n1 + n2;
    let mean_u = (n1 * n2) / 2.0;
    let variance_u = (n1 * n2 * (total + 1.0)) / 12.0;

    let z = (u_statistic - mean_u) / variance_u.sqrt();

    let dist = Normal::new(0.0, 1.0).unwrap();

    let p_value = calculate_p(z, tail_type, &dist);

    // Determine whether to reject the null hypothesis
    let reject_null = p_value < alpha;

    Ok(TestResult {
        test_statistic: u_statistic,
        p_value,
        confidence_interval: (f64::NAN, f64::NAN), // Confidence interval not applicable
        null_hypothesis: "H0: The distributions of both groups are equal.".to_string(),
        alt_hypothesis: "Ha: The distributions of both groups are not equal.".to_string(),
        reject_null,
    })
}
