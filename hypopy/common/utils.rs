pub fn mean_null_hypothesis(num_groups: usize) -> String {
    let mut hypothesis = "H0: µ1".to_string();
    for i in 2..=num_groups {
        hypothesis.push_str(&format!(" = µ{}", i));
    }
    hypothesis
}
