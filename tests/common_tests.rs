#[cfg(test)]
mod tests_common {
    use hypors::common::{calculate_ci, calculate_p, TailType};
    use statrs::distribution::StudentsT;

    #[test]
    fn test_calculate_p_left_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = -2.0;
        let p_value = calculate_p(t_stat, TailType::Left, &t_dist); // Pass by reference
        assert!(p_value < 0.05); // Assuming alpha is 0.05 for this test
    }

    #[test]
    fn test_calculate_p_right_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Right, &t_dist); // Pass by reference
        assert!(p_value < 0.05);
    }

    #[test]
    fn test_calculate_p_two_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Two, &t_dist); // Pass by reference
        assert!(p_value < 0.10); // Since it's two-tailed, we expect the p-value to be larger than one-tailed
    }

    #[test]
    fn test_calculate_ci() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let mean = 5.0;
        let std_error = 1.0;
        let alpha = 0.05;
        let ci = calculate_ci(mean, std_error, alpha, &t_dist); // Pass by reference
        assert!(ci.0 < mean && ci.1 > mean); // Check that the mean lies within the confidence interval
    }
}
