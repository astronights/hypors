//! # Analysis of Variance (ANOVA) Tests
//!
//! This module provides functionality for conducting statistical tests related to Analysis of Variance (ANOVA).
//! It specifically implements one-way ANOVA to test for significant differences between group means.
//!
//! ## Submodules
//!
//! - `one_way`: Contains functions for performing one-way ANOVA tests.
//! - `sample_size`: Contains functions for calculating the required sample size for ANOVA tests.
//!
//! ## Usage
//! To perform a one-way ANOVA test, you can use the `anova` function. The function accepts a slice of data groups,
//! where each group is represented as a `Series`.
//!
//! ### Sample Size Calculation
//! To calculate the required sample size for an ANOVA test, use the `f_sample_size` function. This function requires
//! parameters such as significance level (alpha), power, number of groups, and effect size.
//!
//! ## Example
//! ```rust
//! use crate::anova::{anova, f_sample_size};
//! use polars::prelude::*;
//!
//! // Sample data groups
//! let group1 = Series::new("Group 1", vec![2.0, 3.0, 3.0, 5.0, 6.0]);
//! let group2 = Series::new("Group 2", vec![3.0, 4.0, 4.0, 6.0, 8.0]);
//! let group3 = Series::new("Group 3", vec![5.0, 6.0, 7.0, 8.0, 9.0]);
//!
//! // Perform one-way ANOVA
//! let result = anova(&[&group1, &group2, &group3], 0.05).unwrap();
//! println!("F-statistic: {}, p-value: {}", result.test_statistic, result.p_value);
//!
//! // Calculate required sample size for ANOVA
//! let alpha = 0.05; // significance level
//! let power = 0.80; // desired power
//! let num_groups = 3; // number of groups
//! let effect_size = 0.25; // expected effect size
//! let sample_size = f_sample_size(alpha, power, num_groups, effect_size);
//! println!("Required sample size: {}", sample_size);
//! ```
//!
//! ## Notes
//! This module relies on the Polars crate for data manipulation. Make sure to handle missing data appropriately
//! before running the tests.

pub mod one_way;
pub mod sample_size;

pub use one_way::anova;
pub use sample_size::f_sample_size;
