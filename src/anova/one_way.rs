use crate::common::{StatError, TailType, TestResult, calculate_p, mean_null_hypothesis};
use statrs::distribution::FisherSnedecor;

/// Performs a one-way ANOVA test to compare the means of multiple independent groups.
///
/// # Arguments
///
/// * `data_groups` - A slice of data groups, where each group is an iterable of numeric values.
/// * `alpha` - Significance level (e.g., 0.05).
///
/// # Returns
///
/// A `Result<TestResult, StatError>` with F-statistic, p-value, hypotheses, and rejection status.
///
/// # Errors
///
/// Returns `StatError` if:
/// - There are fewer than 2 groups
/// - Any group is empty
/// - Statistical computation fails
///
/// # Example
///
/// ```rust
/// use hypors::anova::anova;
/// let g1 = vec![2.0, 3.0, 3.0, 5.0, 6.0];
/// let g2 = vec![3.0, 4.0, 4.0, 6.0, 8.0];
/// let g3 = vec![5.0, 6.0, 7.0, 8.0, 9.0];
///
/// let groups = vec![&g1, &g2, &g3];
/// let result = anova(&groups, 0.05).unwrap();
/// assert!(result.p_value > 0.0 && result.p_value < 1.0);
/// ```
pub fn anova<T, I>(data_groups: &[I], alpha: f64) -> Result<TestResult, StatError>
where
    T: Into<f64> + Copy,
    I: AsRef<[T]>,
{
    let num_groups = data_groups.len();
    if num_groups < 2 {
        return Err(StatError::ComputeError(
            "ANOVA requires at least two groups".into(),
        ));
    }

    // Flatten all data and compute grand mean
    let mut all_values = Vec::new();
    for group in data_groups {
        let slice = group.as_ref();
        if slice.is_empty() {
            return Err(StatError::EmptyData);
        }
        all_values.extend(slice.iter().copied().map(Into::into));
    }

    let total_n = all_values.len() as f64;
    let grand_mean = all_values.iter().sum::<f64>() / total_n;

    // Sum of Squares Between Groups (SSB)
    let ss_between = data_groups.iter().fold(0.0, |acc, group| {
        let values: Vec<f64> = group.as_ref().iter().copied().map(Into::into).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        acc + n * (mean - grand_mean).powi(2)
    });

    // Sum of Squares Within Groups (SSW)
    let ss_within = data_groups.iter().fold(0.0, |acc, group| {
        let values: Vec<f64> = group.as_ref().iter().copied().map(Into::into).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        acc + values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
    });

    let df_between = (num_groups - 1) as f64;
    let df_within = total_n - num_groups as f64;

    if df_within <= 0.0 {
        return Err(StatError::ComputeError(
            "Degrees of freedom too small".into(),
        ));
    }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    if ms_within == 0.0 {
        return Err(StatError::ComputeError(
            "Mean square within groups is zero".into(),
        ));
    }

    let f_statistic = ms_between / ms_within;

    let f_dist = FisherSnedecor::new(df_between, df_within)
        .map_err(|e| StatError::ComputeError(format!("Failed to create F distribution: {e}")))?;

    let p_value = calculate_p(f_statistic, TailType::Right, &f_dist);
    let reject_null = p_value < alpha;

    let null_hypothesis = mean_null_hypothesis(num_groups);
    let alt_hypothesis = "Ha: At least one group mean is different".to_string();

    Ok(TestResult {
        test_statistic: f_statistic,
        p_value,
        reject_null,
        null_hypothesis,
        alt_hypothesis,
        confidence_interval: (f64::NAN, f64::NAN), // Not applicable for ANOVA
    })
}
