#[cfg(test)]
mod tests_common {
    use hypors::common::{calculate_chi2_ci, calculate_ci, calculate_p, TailType};
    use statrs::distribution::{ChiSquared, StudentsT};

    #[test]
    fn test_calculate_p_left_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = -2.0;
        let p_value = calculate_p(t_stat, TailType::Left, &t_dist);
        assert!(p_value < 0.05);
    }

    #[test]
    fn test_calculate_p_right_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Right, &t_dist);
        assert!(p_value < 0.05);
    }

    #[test]
    fn test_calculate_p_two_tail() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let t_stat = 2.0;
        let p_value = calculate_p(t_stat, TailType::Two, &t_dist);
        assert!(p_value < 0.10);
    }

    #[test]
    fn test_calculate_ci() {
        let t_dist = StudentsT::new(0.0, 1.0, 10.0).unwrap();
        let mean = 5.0;
        let std_error = 1.0;
        let alpha = 0.05;
        let ci = calculate_ci(mean, std_error, alpha, &t_dist);
        assert!(ci.0 < mean && ci.1 > mean); // Check that the mean lies within the confidence interval
    }

    #[test]
    fn test_calculate_chi2_ci() {
        let df = 9.0; // Degrees of freedom
        let chi_sq_dist = ChiSquared::new(df).unwrap(); // Chi-Squared distribution
        let sample_variance = 2.5;
        let alpha = 0.05;

        let ci = calculate_chi2_ci(sample_variance, alpha, &chi_sq_dist);
        assert!(ci.0 < sample_variance && ci.1 > sample_variance); // Check that the sample variance lies within the confidence interval
    }
}
