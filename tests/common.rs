#[cfg(test)]
mod tests_common {
    use hypors::common::{
        StatError, TailType, TestResult, calculate_chi2_ci, calculate_ci, calculate_p,
    };
    use statrs::distribution::{ChiSquared, StudentsT};

    // Constants to avoid magic numbers
    const EPSILON: f64 = 1e-4; // For floating-point comparisons

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

    #[test]
    fn test_empty_data_error() {
        let error = StatError::EmptyData;
        assert_eq!(error.to_string(), "Cannot perform test on empty data");
        assert_eq!(format!("{:?}", error), "EmptyData");
    }

    #[test]
    fn test_insufficient_data_error() {
        let error = StatError::InsufficientData;
        assert_eq!(error.to_string(), "Insufficient data for statistical test");
        assert_eq!(format!("{:?}", error), "InsufficientData");
    }

    #[test]
    fn test_compute_error() {
        let msg = "Failed to create distribution".to_string();
        let error = StatError::ComputeError(msg.clone());
        assert_eq!(error.to_string(), format!("Computation error: {}", msg));
        assert_eq!(format!("{:?}", error), format!("ComputeError({:?})", msg));
    }

    #[test]
    fn test_error_equality() {
        let error1 = StatError::EmptyData;
        let error2 = StatError::EmptyData;
        let error3 = StatError::InsufficientData;

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_error_cloning() {
        let original = StatError::ComputeError("Test message".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_result_integration() {
        fn validate_data(data: &[f64]) -> Result<(), StatError> {
            if data.is_empty() {
                return Err(StatError::EmptyData);
            }
            if data.len() < 2 {
                return Err(StatError::InsufficientData);
            }
            Ok(())
        }

        // Test empty data
        assert_eq!(validate_data(&[]), Err(StatError::EmptyData));

        // Test insufficient data
        assert_eq!(validate_data(&[1.0]), Err(StatError::InsufficientData));

        // Test valid data
        assert_eq!(validate_data(&[1.0, 2.0]), Ok(()));
    }

    #[test]
    fn test_error_propagation_with_question_mark() {
        fn process_data(data: &[f64]) -> Result<f64, StatError> {
            if data.is_empty() {
                return Err(StatError::EmptyData);
            }
            if data.len() < 2 {
                return Err(StatError::InsufficientData);
            }
            Ok(data.iter().sum::<f64>() / data.len() as f64)
        }

        fn wrapper_function(data: &[f64]) -> Result<String, StatError> {
            let mean = process_data(data)?; // Uses ? operator
            Ok(format!("Mean: {:.2}", mean))
        }

        // Test error propagation
        assert_eq!(wrapper_function(&[]), Err(StatError::EmptyData));
        assert_eq!(wrapper_function(&[1.0]), Err(StatError::InsufficientData));
        assert_eq!(wrapper_function(&[1.0, 3.0]), Ok("Mean: 2.00".to_string()));
    }
}
