#[cfg(test)]
mod tests_mann_whitney {
    use hypors::common::TailType;
    use hypors::mann_whitney::u_test;

    const EPSILON: f64 = 0.0001; // For floating-point comparisons

    #[test]
    fn test_u_test() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let alpha = 0.05;

        let result = u_test(data1, data2, alpha, TailType::Two).unwrap();

        let expected_u_statistic = 4.5;
        let expected_p_value = 0.0946;
        let expected_null_hypothesis = "H0: The distributions of both groups are equal.";
        let expected_alt_hypothesis = "Ha: The distributions of both groups are not equal.";

        assert!((result.test_statistic - expected_u_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_u_test_equal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.05;

        let result = u_test(data.clone(), data, alpha, TailType::Two).unwrap();

        let expected_u_statistic = 12.5;
        let expected_p_value = 1.0;
        let expected_null_hypothesis = "H0: The distributions of both groups are equal.";
        let expected_alt_hypothesis = "Ha: The distributions of both groups are not equal.";

        assert!((result.test_statistic - expected_u_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }
}
