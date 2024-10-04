#[cfg(test)]
mod tests_common {
    use hypors::common::{calculate_chi2_ci, calculate_ci, calculate_p, TailType, TestResult};
    use statrs::distribution::{ChiSquared, StudentsT};

    // Constants to avoid magic numbers
    const EPSILON: f64 = 1e-6; // For floating-point comparisons

    #[test]
    fn test_calculate_p_left_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = -2.0;
        let p_value = calculate_p(t_stat, TailType::Left, &t_dist);
        let expected_p_value = 0.036694;

        assert!((p_value - expected_p_value).abs() < EPSILON);
    }

    #[test]
    fn test_calculate_p_right_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Right, &t_dist);
        let expected_p_value = 0.036694;

        assert!((p_value - expected_p_value).abs() < EPSILON);
    }

    #[test]
    fn test_calculate_p_two_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Two, &t_dist);
        let expected_p_value = 0.073388;

        assert!((p_value - expected_p_value).abs() < EPSILON);
    }

    #[test]
    fn test_calculate_ci() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let mean = 5.0;
        let std_error = 1.0;
        let alpha = 0.05;
        let ci = calculate_ci(mean, std_error, alpha, &t_dist);

        let expected_ci_lower = 2.771861;
        let expected_ci_upper = 7.228138;

        assert!((ci.0 - expected_ci_lower).abs() < EPSILON);
        assert!((ci.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_calculate_chi2_ci() {
        let df = 9.0;
        let chi_sq_dist = ChiSquared::new(df).unwrap();
        let sample_variance = 2.5;
        let alpha = 0.05;

        let ci = calculate_chi2_ci(sample_variance, alpha, &chi_sq_dist);

        let expected_ci_lower = 0.591388;
        let expected_ci_upper = 4.166106;

        assert!((ci.0 - expected_ci_lower).abs() < EPSILON);
        assert!((ci.1 - expected_ci_upper).abs() < EPSILON);
    }

    #[test]
    fn test_tail_type() {
        assert_eq!(TailType::Left, TailType::Left);
        assert_eq!(TailType::Right, TailType::Right);
        assert_eq!(TailType::Two, TailType::Two);
    }

    #[test]
    fn test_test_result() {
        let t_stat = 2.0;
        let p_value = 0.036;
        let confidence_interval = (4.0, 6.0);
        let null_hypothesis = "H0";
        let alt_hypothesis = "Ha";
        let reject_null = false;
        let result = TestResult {
            test_statistic: t_stat,
            p_value,
            confidence_interval,
            null_hypothesis: null_hypothesis.to_string(),
            alt_hypothesis: alt_hypothesis.to_string(),
            reject_null,
        };

        assert_eq!(result.test_statistic, t_stat);
        assert_eq!(result.p_value, p_value);
        assert_eq!(result.confidence_interval, confidence_interval);
        assert_eq!(result.null_hypothesis, null_hypothesis);
        assert_eq!(result.alt_hypothesis, alt_hypothesis);
        assert_eq!(result.reject_null, reject_null);
    }
}
