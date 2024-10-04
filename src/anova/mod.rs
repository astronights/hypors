//! # Analysis of Variance (ANOVA) Tests
//!
//! This module provides functionality for conducting statistical tests related to Analysis of Variance (ANOVA).
//! It specifically implements one-way ANOVA to test for significant differences between group means.
//!
//! ## Submodules
//!
//! - `one_way`: Contains functions for performing one-way ANOVA tests.
//!
//! ## Usage
//! To perform a one-way ANOVA test, you can use the `anova` function. The function accepts a slice of data groups,
//! where each group is represented as a `Series`.
//!
//! ## Example
//! ```rust
//! use crate::anova::anova;
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
//! ```
//!
//! ## Notes
//! This module relies on the Polars crate for data manipulation. Make sure to handle missing data appropriately
//! before running the tests.

pub mod one_way;

pub use one_way::anova;
