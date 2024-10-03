#[cfg(test)]
mod tests_t_test {
    use hypors::common::TailType;
    use hypors::t_test::{one_sample, two_sample_ind, two_sample_paired};
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_one_sample() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_mean = 5.0;
        let alpha = 0.05;

        // Perform one-sample t-test (two-tailed)
        let result = one_sample(&data, pop_mean, TailType::Two, alpha).unwrap();
        
        let expected_t_statistic = 0.419;
        let expected_p_value = 0.696; 

        // Check the t-statistic
        assert!((result.test_statistic - expected_t_statistic).abs() < 0.001);
        
        // Check the p-value
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the reject_null flag (tweak this based on expected results)
        assert_eq!(result.reject_null, false); // Based on data analysis
    }

    #[test]
    fn test_two_sample_paired() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let alpha = 0.05;

        let result = two_sample_paired(&data1, &data2, TailType::Two, alpha).unwrap();
        
        // Calculate expected values based on known statistics
        let expected_t_statistic = 0.598;
        let expected_p_value = 0.582;

        // Check the t-statistic and p-value
        assert!((result.test_statistic - expected_t_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }

    #[test]
    fn test_two_sample_ind_unpooled() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let alpha = 0.05;

        // Perform two-sample independent t-test (unpooled)
        let result = two_sample_ind(&data1, &data2, TailType::Two, alpha, false).unwrap();
        
        // Calculate expected values based on known statistics
        let expected_t_statistic = 0.099;
        let expected_p_value = 0.922; 

        // Check the t-statistic and p-value
        assert!((result.test_statistic - expected_t_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }

    #[test]
    fn test_two_sample_ind_pooled() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let alpha = 0.05;

        // Perform two-sample independent t-test (pooled)
        let result = two_sample_ind(&data1, &data2, TailType::Two, alpha, true).unwrap();
        
        // Calculate expected values based on known statistics
        let expected_t_statistic = 0.099;
        let expected_p_value = 0.922;

        // Check the t-statistic and p-value
        assert!((result.test_statistic - expected_t_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }
}
