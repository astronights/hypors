#[cfg(test)]
mod tests_z_test {
    use hypors::common::TailType;
    use hypors::z::{z_sample_size, z_test, z_test_ind, z_test_paired};
    use polars::prelude::*;

    const EPSILON: f64 = 0.001; // For floating-point comparisons

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_z_test() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_mean = 5.0;
        let pop_std = 2.0;
        let alpha = 0.05;

        let result = z_test(&data, pop_mean, pop_std, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.670;
        let expected_p_value = 0.502;
        let expected_ci_lower = 3.846954;
        let expected_ci_upper = 7.353045;
        let expected_null_hypothesis = "H0: µ = 5";
        let expected_alt_hypothesis = "Ha: µ ≠ 5";

        assert!((result.test_statistic - expected_z_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, false);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);
    }

    #[test]
    fn test_z_test_paired() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let pop_std_diff = 1.5;
        let alpha = 0.05;

        let result = z_test_paired(&data1, &data2, pop_std_diff, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.298;
        let expected_p_value = 0.765;
        let expected_ci_lower = -1.114783;
        let expected_ci_upper = 1.514783;
        let expected_null_hypothesis = "H0: µ1 = µ2";
        let expected_alt_hypothesis = "Ha: µ1 ≠ µ2";

        assert!((result.test_statistic - expected_z_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, false);

        println!("{} {}", result.confidence_interval.0, expected_ci_lower);
        println!("{} {}", result.confidence_interval.1, expected_ci_upper);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);
    }

    #[test]
    fn test_z_test_ind() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let pop_std1 = 2.0;
        let pop_std2 = 1.5;
        let alpha = 0.05;

        let result = z_test_ind(&data1, &data2, pop_std1, pop_std2, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.179;
        let expected_p_value = 0.858;
        let expected_ci_lower = -1.991306;
        let expected_ci_upper = 2.391306;
        let expected_null_hypothesis = "H0: µ1 = µ2";
        let expected_alt_hypothesis = "Ha: µ1 ≠ µ2";

        assert!((result.test_statistic - expected_z_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);
        assert_eq!(result.reject_null, false);

        assert!((result.confidence_interval.0 - expected_ci_lower).abs() < EPSILON);
        assert!((result.confidence_interval.1 - expected_ci_upper).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);
    }

    #[test]
    fn test_z_sample_size() {
        let effect_size = 0.3;
        let alpha = 0.05;
        let power = 0.80;
        let std_dev = 1.0;
        let tail = TailType::Two;

        let n = z_sample_size(effect_size, alpha, power, std_dev, tail);

        let expected_sample_size = 87.79;

        assert!(
            (n - expected_sample_size).abs() < 1.0,
            "Sample size is incorrect"
        );
    }
}
