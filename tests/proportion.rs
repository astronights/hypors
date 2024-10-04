#[cfg(test)]
mod tests_proportion {
    use hypors::common::TailType;
    use hypors::proportion::{z_test, z_test_ind};
    use polars::prelude::*;

    const EPSILON: f64 = 0.001; // Tolerance for floating-point comparisons

    // Helper function to create a Polars Series
    fn create_series(data: Vec<u32>) -> Series {
        Series::new("data".into(), data)
    }

    #[test]
    fn test_z_test() {
        let data = create_series(vec![1, 1, 1, 0, 0]);
        let null_prop = 0.5;
        let alpha = 0.05;

        let result = z_test(&data, null_prop, TailType::Two, alpha).unwrap();

        let expected_z_statistic = 0.447;
        let expected_p_value = 0.655;
        let expected_null_hypothesis = "H0: p = 0.5";
        let expected_alt_hypothesis = "Ha: p ≠ 0.5";

        assert!((result.test_statistic - expected_z_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_z_test_ind_unpooled() {
        let data1 = create_series(vec![1, 1, 1, 0, 0]);
        let data2 = create_series(vec![1, 1, 0, 0, 0]);
        let alpha = 0.05;

        let result = z_test_ind(&data1, &data2, TailType::Two, alpha, false).unwrap();

        let expected_z_statistic = 0.645;
        let expected_p_value = 0.518;
        let expected_null_hypothesis = "H0: p1 = p2";
        let expected_alt_hypothesis = "Ha: p1 ≠ p2";

        assert!((result.test_statistic - expected_z_statistic).abs() < EPSILON);
        assert!((result.p_value - expected_p_value).abs() < EPSILON);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }

    #[test]
    fn test_z_test_ind_pooled() {
        let data1 = create_series(vec![1, 1, 1, 0, 0]);
        let data2 = create_series(vec![1, 1, 0, 0, 0]);
        let alpha = 0.05;

        let result = z_test_ind(&data1, &data2, TailType::Two, alpha, true).unwrap();

        let expected_z_statistic = 0.632;
        let expected_p_value = 0.527;
        let expected_null_hypothesis = "H0: p1 = p2";
        let expected_alt_hypothesis = "Ha: p1 ≠ p2";

        assert!((result.test_statistic - expected_z_statistic).abs() < 0.001);
        assert!((result.p_value - expected_p_value).abs() < 0.001);

        assert_eq!(result.null_hypothesis, expected_null_hypothesis);
        assert_eq!(result.alt_hypothesis, expected_alt_hypothesis);

        assert_eq!(result.reject_null, false);
    }
}
