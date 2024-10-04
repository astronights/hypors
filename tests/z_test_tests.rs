#[cfg(test)]
mod tests_z_test {
    use hypors::common::TailType;
    use hypors::z_test::{one_sample, two_sample_ind, two_sample_paired};
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_one_sample() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_mean = 5.0;
        let pop_std = 2.0;
        let alpha = 0.05;

        // Perform one-sample z-test (two-tailed)
        let result = one_sample(&data, pop_mean, pop_std, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.670;
        let expected_p_value = 0.502;

        // Check the z-statistic
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);

        // Check the p-value
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the reject_null flag
        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_two_sample_paired() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let pop_std_diff = 1.5; // Population standard deviation of differences
        let alpha = 0.05;

        // Perform paired two-sample z-test (two-tailed)
        let result = two_sample_paired(&data1, &data2, pop_std_diff, TailType::Two, alpha).unwrap();

        // Calculate expected values based on known statistics
        let expected_z_statistic = 0.298;
        let expected_p_value = 0.765;

        // Check the z-statistic and p-value
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the null hypothesis
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }

    #[test]
    fn test_two_sample_ind() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let pop_std1 = 2.0; // Population std for sample 1
        let pop_std2 = 1.5; // Population std for sample 2
        let alpha = 0.05;

        // Perform two-sample independent z-test (unpooled variances)
        let result =
            two_sample_ind(&data1, &data2, pop_std1, pop_std2, TailType::Two, alpha).unwrap();

        // Calculate expected values based on known statistics
        let expected_z_statistic = 0.179;
        let expected_p_value = 0.858;

        // Check the z-statistic and p-value
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the null hypothesis
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }
}
