#[cfg(test)]
mod tests_anova {
    use hypors::anova::one_way;
    use polars::prelude::*;

    // Helper function to create a Polars Series
    fn create_series(data: Vec<f64>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_one_way() {
        let data1 = create_series(vec![2.0, 3.0, 3.0, 5.0, 6.0]);
        let data2 = create_series(vec![3.0, 4.0, 4.0, 6.0, 8.0]);
        let data3 = create_series(vec![5.0, 6.0, 7.0, 8.0, 9.0]);

        let result = one_way(&[&data1, &data2, &data3], 0.05).unwrap();

        let expected_f_statistic = 4.261;
        let expected_p_value = 0.039;

        // Check the F-statistic
        assert!((result.test_statistic - expected_f_statistic).abs() < 0.001);

        // Check the p-value
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        assert_eq!(result.reject_null, true);

        // Check the null hypothesis string
        assert_eq!(result.null_hypothesis, "H0: µ1 = µ2 = µ3");
    }
}
