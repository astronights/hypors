#[cfg(test)]
mod tests {
    use hypors::t_test::{calculate_ci, calculate_p, TailType};
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data", data)
    }

    #[test]
    fn test_one_sample() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_mean = 5.0;
        let alpha = 0.05;

        // Perform one-sample t-test (two-tailed)
        let result = one_sample(&data, pop_mean, TailType::Two, alpha).unwrap();
        
        // Check t-statistic is computed
        assert!(result.t_stat != 0.0);
        
        // Check p-value is reasonable
        assert!(result.p_value < 1.0 && result.p_value > 0.0);

        // Check the reject_null flag (you can fine-tune this based on data)
        assert_eq!(result.reject_null, false);  // Since data seems close to population mean
    }

    #[test]
    fn test_two_sample_paired() {
        let data1 = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let data2 = create_series(vec![1.0, 3.0, 6.0, 7.0, 10.0]);
        let alpha = 0.05;

        // Perform two-sample paired t-test
        let result = two_sample_paired(&data1, &data2, TailType::Two, alpha).unwrap();
        
        // Check the t-statistic and p-value
        assert!(result.t_stat != 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);

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
        
        // Check the t-statistic and p-value
        assert!(result.t_stat != 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);

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
        
        // Check the t-statistic and p-value
        assert!(result.t_stat != 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2");
    }
}
