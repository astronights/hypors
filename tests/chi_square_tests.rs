#[cfg(test)]
mod tests_chi_square {
    use hypors::chi_square::{goodness_of_fit, independence, variance}; // Ensure you import the Chi-Square module
    use hypors::common::TailType;
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_variance() {
        let data = create_series(vec![2.0, 3.0, 5.0, 7.0, 11.0]);
        let pop_variance = 5.0;
        let alpha = 0.05;

        // Perform Chi-Square Test for Variance
        let result = variance(&data, pop_variance, TailType::Two, alpha).unwrap();

        let expected_chi_square_stat = 10.24;
        let expected_p_value = 0.073;

        // Check the test statistic and p-value
        assert!((result.test_statistic - expected_chi_square_stat).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check the reject_null flag
        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_independence() {
        let contingency_table = vec![vec![20.0, 30.0], vec![50.0, 10.0]];
        let alpha = 0.05;

        // Perform Chi-Square Test for Independence
        let result = independence(&contingency_table, alpha).unwrap();

        let expected_chi_square_stat = 22.131;
        let expected_p_value = 0.000;

        // Check the test statistic and p-value
        assert!((result.test_statistic - expected_chi_square_stat).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(result.null_hypothesis, "H0: Variables are independent");
    }

    #[test]
    fn test_goodness_of_fit() {
        let observed = create_series(vec![30.0, 10.0, 20.0]);
        let expected = create_series(vec![25.0, 15.0, 20.0]);
        let alpha = 0.05;

        // Perform Chi-Square Goodness of Fit Test
        let result = goodness_of_fit(&observed, &expected, alpha).unwrap();

        let expected_chi_square_stat = 2.666;
        let expected_p_value = 0.263;

        // Check the test statistic and p-value
        assert!((result.test_statistic - expected_chi_square_stat).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        // Check that the null hypothesis is correctly formed
        assert_eq!(
            result.null_hypothesis,
            "H0: Observed distribution matches expected distribution"
        );
    }
}
