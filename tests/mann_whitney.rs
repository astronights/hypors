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

        // scipy.stats.mannwhitneyu([1,2,3,4,5], [3,4,5,6,7],
        // alternative="two-sided", method="asymptotic") -> p = 0.1138462980
        let expected_u_statistic = 4.5;
        let expected_p_value = 0.1138463;
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

    #[test]
    fn test_u_test_one_sided_direction() {
        // Group 1 is clearly larger: Right (greater) must be significant,
        // Left (less) must not — one-sided tests are direction-sensitive.
        let larger = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let smaller = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.05;

        // scipy.stats.mannwhitneyu([5..9], [1..5], alternative="greater",
        // method="asymptotic") -> p = 0.0079853482
        let right = u_test(larger.clone(), smaller.clone(), alpha, TailType::Right).unwrap();
        assert!((right.p_value - 0.0079853).abs() < EPSILON);
        assert!(right.reject_null);

        // scipy.stats.mannwhitneyu([5..9], [1..5], alternative="less",
        // method="asymptotic") -> p = 0.9955920709
        let left = u_test(larger, smaller, alpha, TailType::Left).unwrap();
        assert!((left.p_value - 0.9955921).abs() < EPSILON);
        assert!(!left.reject_null);
    }

    #[test]
    fn test_u_test_one_sided_overlapping() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let alpha = 0.05;

        // scipy.stats.mannwhitneyu([1..5], [3..7], alternative="greater",
        // method="asymptotic") -> p = 0.9634301002
        let right = u_test(data1.clone(), data2.clone(), alpha, TailType::Right).unwrap();
        assert!((right.p_value - 0.9634301).abs() < EPSILON);

        // scipy.stats.mannwhitneyu([1..5], [3..7], alternative="less",
        // method="asymptotic") -> p = 0.0569231490
        let left = u_test(data1, data2, alpha, TailType::Left).unwrap();
        assert!((left.p_value - 0.0569231).abs() < EPSILON);
    }

    #[test]
    fn test_u_test_heavy_ties() {
        // Ties shrink the variance; the tie-corrected sigma must be used.
        let data1 = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let data2 = vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let alpha = 0.05;

        let result = u_test(data1, data2, alpha, TailType::Two).unwrap();

        // scipy.stats.mannwhitneyu([1,2,2,3,3,3], [2,3,3,4,4,5],
        // alternative="two-sided", method="asymptotic")
        // -> U1 = 7.0, p = 0.0784029345
        let expected_u_statistic = 7.0; // min(U1, U2) = min(7, 29)
        let expected_p_value = 0.0784029;

        assert!((result.test_statistic - expected_u_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_u_test_all_tied() {
        // Zero variance (every observation identical): scipy
        // propagates NaN (mannwhitneyu([2,2,2], [2,2,2],
        // method="asymptotic") -> statistic 4.5, pvalue nan), and a
        // NaN p-value never rejects the null.
        let result = u_test(
            vec![2.0, 2.0, 2.0],
            vec![2.0, 2.0, 2.0],
            0.05,
            TailType::Two,
        )
        .unwrap();
        assert!((result.test_statistic - 4.5).abs() < EPSILON);
        assert!(result.p_value.is_nan());
        assert!(!result.reject_null);

        // One-sided tails subtract a signed 0.5, so their z is an
        // infinity rather than 0/0: scipy 1.18 gives p = 1.0.
        for tail in [TailType::Right, TailType::Left] {
            let result = u_test(vec![2.0, 2.0, 2.0], vec![2.0, 2.0, 2.0], 0.05, tail).unwrap();
            assert!((result.p_value - 1.0).abs() < EPSILON);
            assert!(!result.reject_null);
        }
    }

    #[test]
    fn test_u_test_empty_group() {
        let empty: Vec<f64> = vec![];
        let result = u_test(empty, vec![1.0, 2.0], 0.05, TailType::Two);
        assert!(result.is_err());
    }
}
