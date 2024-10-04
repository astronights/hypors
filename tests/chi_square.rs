#[cfg(test)]
mod tests_chi_square {
    use hypors::chi_square::{goodness_of_fit, independence, variance};
    use hypors::common::TailType;
    use polars::prelude::*;

    const EPSILON: f64 = 0.001; // Tolerance for floating-point comparisons

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_variance() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_variance = 5.0;
        let alpha = 0.05;

        let result = variance(&data, pop_variance, TailType::Two, alpha).unwrap();

        let expected_chi_square_stat = 10.24;
        let expected_p_value = 0.073;
        let expected_null_hypothesis = "H0: σ² = 5";
        let expected_alt_hypothesis = "Ha: σ² ≠ 5";

        assert!((result.test_statistic - expected_chi_square_stat).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_independence() {
        let contingency_table = vec![vec![20.0, 30.0], vec![50.0, 10.0]];
        let alpha = 0.05;

        let result = independence(&contingency_table, alpha).unwrap();

        let expected_chi_square_stat = 22.131;
        let expected_p_value = 0.000;
        let expected_null_hypothesis = "H0: Variables are independent";
        let expected_alt_hypothesis = "Ha: Variables are not independent";

        assert!((result.test_statistic - expected_chi_square_stat).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, true);
    }

    #[test]
    fn test_goodness_of_fit() {
        let observed = create_series(vec![30.0, 10.0, 20.0]);
        let expected = create_series(vec![25.0, 15.0, 20.0]);
        let alpha = 0.05;

        let result = goodness_of_fit(&observed, &expected, alpha).unwrap();

        let expected_chi_square_stat = 2.666;
        let expected_p_value = 0.263;
        let expected_null_hypothesis = "H0: Observed distribution matches expected distribution";
        let expected_alt_hypothesis =
            "Ha: Observed distribution does not match expected distribution";

        assert!((result.test_statistic - expected_chi_square_stat).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);

    }
}
