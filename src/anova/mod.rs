//! # Analysis of Variance (ANOVA) Tests
//!
//! The `anova` module provides functionality for performing one-way ANOVA tests.
//!
//! ANOVA (Analysis of Variance) is a statistical method used to compare means
//! across multiple groups to determine if at least one group mean is significantly
//! different from the others. It is particularly useful when dealing with three or more groups.
//!
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for ANOVA tests, you can use the following function:
//! - `f_sample_size`: Calculates the necessary sample size for one-way ANOVA tests
//!
//! ## Submodules
//!
//! - `one_way`: Contains functions for performing one-way ANOVA tests.
//! - `sample_size`: Contains functions for calculating the required sample size for ANOVA tests.
//!
//! ## Exports
//!
//! The following functions are made available for use:
//! - `anova`: Performs one-way ANOVA tests on multiple groups of data.
//! - `f_sample_size`: Calculates the required sample size for one-way ANOVA tests
//!
//! ## Example
//! ```rust
//! use hypors::anova::{anova, f_sample_size};
//! ```

pub mod one_way;
pub mod sample_size;

pub use one_way::anova;
pub use sample_size::f_sample_size;
