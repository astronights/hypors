use crate::common::{calculate_p, mean_null_hypothesis, TailType, TestResult};
use polars::prelude::*;
use pyo3::prelude::*;
use statrs::distribution::FisherSnedecor;
use std::f64;

/// Perform a one-way Analysis of Variance (ANOVA) test to compare the means of multiple groups.
///
/// One-way ANOVA tests whether there are statistically significant differences between the means
/// of two or more independent groups. The null hypothesis (H0) assumes that all group means are equal.
///
/// ## Arguments
///
/// * `data_groups` - A slice of `Series` where each `Series` represents the data for a different independent group.
/// * `alpha` - Significance level for the test (e.g., 0.05).
///
/// ## Returns
///
/// This function returns a `Result<TestResult, PyErr>`, where:
/// - `TestResult` contains:
///     * `test_statistic` - The F-statistic value computed from the data.
///     * `p_value` - The p-value associated with the test statistic, used to determine the statistical significance.
///     * `reject_null` - A boolean indicating whether to reject the null hypothesis (`true` if p-value < alpha).
///     * `null_hypothesis` - A string representing the null hypothesis (H0: All group means are equal).
///     * `alt_hypothesis` - A string representing the alternative hypothesis (H1: At least one group mean is different).
///     * `confidence_interval` - ANOVA does not produce a traditional confidence interval, so this will contain `(NaN, NaN)`.
///
/// ## Example
///
/// ```rust
/// use hypors::anova::anova;
/// use polars::prelude::*;
///
/// // Define data for three independent groups
/// let data1 = Series::new("Group1".into(), vec![2.0, 3.0, 3.0, 5.0, 6.0]);
/// let data2 = Series::new("Group2".into(), vec![3.0, 4.0, 4.0, 6.0, 8.0]);
/// let data3 = Series::new("Group3".into(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // Perform ANOVA
/// let result = anova(&[&data1, &data2, &data3], 0.05).unwrap();
///
/// // Check the results
/// println!("F-statistic: {}", result.test_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Reject Null Hypothesis: {}", result.reject_null);
/// ```
///
/// In this example, three groups are tested using one-way ANOVA to determine whether their means are statistically different.
///
/// ## Notes
/// - The p-value is computed using the Fisher-Snedecor distribution (F-distribution).
/// - If the p-value is less than the significance level (alpha), the null hypothesis is rejected, suggesting that not all group means are equal.
///
fn anova(data_groups: &[&Series], alpha: f64) -> PyResult<TestResult> {
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

#[pyfunction]
pub fn py_anova(data_groups: Vec<Py<Series>>, alpha: f64) -> PyResult<TestResult> {
    Python::with_gil(|py| {
        // Borrow the Series while holding the GIL
        let groups: Vec<Result<Series, PyErr>> = data_groups
            .into_iter()
            .map(|s| s.extract::<Series>(py))
            .collect();

        let groups = groups?; // Propagate any extraction errors

        // Now you have a Vec<Series> which can be used in your ANOVA function.
        let groups_slice: Vec<&Series> = groups.iter().collect(); // Create a slice of references to Series

        // Call the ANOVA function with the slice
        anova(groups_slice, alpha)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    })
}
