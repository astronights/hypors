#[cfg(test)]
mod tests_anova {
    use hypors::anova::anova;
    use polars::prelude::*;

    const EPSILON: f64 = 0.001; // Tolerance for floating-point comparisons

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_anova() {
        let data1 = create_series(vec![2.0, 3.0, 3.0, 5.0, 6.0]);
        let data2 = create_series(vec![3.0, 4.0, 4.0, 6.0, 8.0]);
        let data3 = create_series(vec![5.0, 6.0, 7.0, 8.0, 9.0]);

        let result = anova(&[&data1, &data2, &data3], 0.05).unwrap();

        let expected_f_statistic = 4.261;
        let expected_p_value = 0.039;
        let expected_null_hypothesis = "H0: µ1 = µ2 = µ3";
        let expected_alt_hypothesis = "Ha: At least one group mean is different";

        assert!((result.test_statistic - expected_f_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, true);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);
    }

    #[test]
    fn test_anova_no_rejection() {
        let data1 = create_series(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let data2 = create_series(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let data3 = create_series(vec![4.0, 5.0, 6.0, 7.0, 8.0]);

        let result = anova(&[&data1, &data2, &data3], 0.05).unwrap();

        let expected_f_statistic = 2.0;
        let expected_p_value = 0.177;
        let expected_null_hypothesis = "H0: µ1 = µ2 = µ3";
        let expected_alt_hypothesis = "Ha: At least one group mean is different";

        assert!((result.test_statistic - expected_f_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, false);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);
    }
}
