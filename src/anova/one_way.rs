use crate::common::{calculate_p, mean_null_hypothesis, TailType, TestResult};
use polars::prelude::*;
use statrs::distribution::FisherSnedecor;
use std::f64;

/// Perform one-way ANOVA test.
///
/// # Arguments
///
/// * `data_groups` - A slice of Series, where each Series represents the data for an independent group.
///
/// # Returns
///
/// Returns `Result<TestResult, PolarsError>`, where `TestResult` contains the F-statistic, p-value, and the decision to reject the null hypothesis.
///
/// The null hypothesis (H0) is that all group means are equal (H0: µ1 = µ2 = µ3 = ...).
///
pub fn anova(data_groups: &[&Series], alpha: f64) -> Result<TestResult, PolarsError> {
    let num_groups = data_groups.len();
    let total_n = data_groups.iter().map(|s| s.len()).sum::<usize>() as f64;

    let grand_mean = data_groups
        .iter()
        .flat_map(|s| s.f64().unwrap().into_no_null_iter())
        .sum::<f64>()
        / total_n;

    // Calculate between-group variance (Sum of Squares Between)
    let ss_between = data_groups.iter().fold(0.0, |sum, group| {
        let group_mean = group.mean().unwrap();
        sum + group.len() as f64 * (group_mean - grand_mean).powi(2)
    });

    // Calculate within-group variance (Sum of Squares Within)
    let ss_within = data_groups.iter().fold(0.0, |sum, group| {
        sum + group
            .f64()
            .unwrap()
            .into_no_null_iter()
            .fold(0.0, |acc, value| {
                acc + (value - group.mean().unwrap()).powi(2)
            })
    });

    let df_between = (num_groups - 1) as f64;
    let df_within = total_n - num_groups as f64;

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    let f_statistic = ms_between / ms_within;

    // Fisher-Snedecor distribution for p-value computation
    let f_distribution = FisherSnedecor::new(df_between, df_within).unwrap();
    let p_value = calculate_p(f_statistic, TailType::Right, &f_distribution);

    let reject_null = p_value < alpha;

    // Dynamically create the null hypothesis string (H0)
    let null_hypothesis = mean_null_hypothesis(num_groups);

    // Alternate hypothesis (H1): "At least one group mean is different"
    let alt_hypothesis = "Ha: At least one group mean is different".to_string();

    Ok(TestResult {
        test_statistic: f_statistic,
        p_value,
        reject_null,
        null_hypothesis,
        alt_hypothesis,
        confidence_interval: (f64::NAN, f64::NAN), // ANOVA doesn't produce a typical confidence interval like t-tests
    })
}
