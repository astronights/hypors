#[cfg(test)]
mod tests_proportion {
    use hypors::common::TailType;
    use hypors::proportion::{one_sample, two_sample};
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<u32>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_one_sample() {
        let data = create_series(vec![1, 1, 1, 0, 0]); // 3 successes, 2 failures
        let null_prop = 0.5;
        let alpha = 0.05;

        // Perform one-sample proportion test (two-tailed)
        let result = one_sample(&data, null_prop, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.447;
        let expected_p_value = 0.655;

        // Check the z-statistic
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);

        // Check the p-value
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the reject_null flag (tweak this based on expected results)
        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_two_sample_unpooled() {
        let data1 = create_series(vec![1, 1, 1, 0, 0]); // 3 successes, 2 failures
        let data2 = create_series(vec![1, 1, 0, 0, 0]); // 2 successes, 3 failures
        let alpha = 0.05;

        // Perform two-sample independent proportion test (unpooled)
        let result = two_sample(&data1, &data2, TailType::Two, alpha, false).unwrap();

        let expected_z_statistic = 0.645;
        let expected_p_value = 0.518;

        // Check the z-statistic and p-value
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: p1 = p2");
    }

    #[test]
    fn test_two_sample_pooled() {
        let data1 = create_series(vec![1, 1, 1, 0, 0]); // 3 successes, 2 failures
        let data2 = create_series(vec![1, 1, 0, 0, 0]); // 2 successes, 3 failures
        let alpha = 0.05;

        // Perform two-sample independent proportion test (pooled)
        let result = two_sample(&data1, &data2, TailType::Two, alpha, true).unwrap();

        let expected_z_statistic = 0.632; 
        let expected_p_value = 0.527;

        // Check the z-statistic and p-value
        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: p1 = p2");
    }

}
