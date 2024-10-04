/// Helper function to dynamically create the null hypothesis string for a given number of groups.
///
/// This function generates a null hypothesis of the form "H0: µ1 = µ2 = ... = µn",
/// indicating that the means of the specified number of groups are equal.
///
/// # Arguments
///
/// * `num_groups` - The number of groups being tested.
///
/// # Returns
///
/// A string representing the null hypothesis for the means of the groups.
///
/// # Example
///
/// ```rust
/// let hypothesis = mean_null_hypothesis(3);
/// assert_eq!(hypothesis, "H0: µ1 = µ2 = µ3");
/// ```
pub fn mean_null_hypothesis(num_groups: usize) -> String {
    let mut hypothesis = "H0: µ1".to_string();
    for i in 2..=num_groups {
        hypothesis.push_str(&format!(" = µ{}", i));
    }
    hypothesis
}
