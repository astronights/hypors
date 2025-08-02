#[cfg(test)]
mod tests_t_test {
    use hypors::common::TailType;
    use hypors::t::{t_sample_size, t_test, t_test_ind, t_test_paired};

    const EPSILON: f64 = 0.001; // For floating-point comparisons

    #[test]
    fn test_t_test() {
        let data = vec![2.0, 3.0, 5.0, 7.0, 11.0];
        let pop_mean = 5.0;
        let alpha = 0.05;

        let result = t_test(data, pop_mean, TailType::Two, alpha).unwrap();

        let expected_t_statistic = 0.374;
        let expected_p_value = 0.726;
        let expected_ci_lower = 1.157687;
        let expected_ci_upper = 10.042312;
        let expected_null_hypothesis = "H0: µ = 5";
        let expected_alt_hypothesis = "Ha: µ ≠ 5";

        assert!((result.test_statistic - expected_t_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, false);
        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_t_test_paired() {
        let data1 = vec![2.0, 3.0, 5.0, 7.0, 11.0];
        let data2 = vec![1.0, 3.0, 6.0, 7.0, 10.0];
        let alpha = 0.05;

        let result = t_test_paired(data1, data2, TailType::Two, alpha).unwrap();

        let expected_t_statistic = 0.534;
        let expected_p_value = 0.621;
        let expected_ci_lower = -0.838850;
        let expected_ci_upper = 1.238850;
        let expected_null_hypothesis = "H0: µ1 = µ2";
        let expected_alt_hypothesis = "Ha: µ1 ≠ µ2";

        assert!((result.test_statistic - expected_t_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_t_test_ind_unpooled() {
        let data1 = vec![2.0, 3.0, 5.0, 7.0, 11.0];
        let data2 = vec![1.0, 3.0, 6.0, 7.0, 10.0];
        let alpha = 0.05;

        let result = t_test_ind(data1, data2, TailType::Two, alpha, false).unwrap();

        let expected_t_statistic = 0.089;
        let expected_p_value = 0.931;
        let expected_ci_lower = -4.967041;
        let expected_ci_upper = 5.367041;
        let expected_null_hypothesis = "H0: µ1 = µ2";
        let expected_alt_hypothesis = "Ha: µ1 ≠ µ2";

        assert!((result.test_statistic - expected_t_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_t_test_ind_pooled() {
        let data1 = vec![2.0, 3.0, 5.0, 7.0, 11.0];
        let data2 = vec![1.0, 3.0, 6.0, 7.0, 10.0];
        let alpha = 0.05;

        let result = t_test_ind(data1, data2, TailType::Two, alpha, true).unwrap();

        let expected_t_statistic = 0.089;
        let expected_p_value = 0.931;
        let expected_ci_lower = -4.966684;
        let expected_ci_upper = 5.366684;
        let expected_null_hypothesis = "H0: µ1 = µ2";
        let expected_alt_hypothesis = "Ha: µ1 ≠ µ2";

        assert!((result.test_statistic - expected_t_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_t_sample_size() {
        let effect_size = 0.3;
        let alpha = 0.05;
        let power = 0.80;
        let std_dev = 1.0;
        let tail = TailType::Two;

        let n = t_sample_size(effect_size, alpha, power, std_dev, tail);
        let expected_sample_size = 88.49;

        assert!(
            (n - expected_sample_size).abs() < 1.0,
            "Sample size is incorrect"
        );
    }
}
