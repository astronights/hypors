//! # Chi Square Tests
//! 
//! The `chi_square` module provides functions for performing Chi-Square tests,
//! including tests for independence in categorical data and tests for variance.
//!
//! The Chi-Square Test for Independence evaluates whether two categorical variables
//! are independent of each other based on a contingency table. The Chi-Square Goodness
//! of Fit Test assesses whether observed frequencies match expected frequencies.
//!
//! The Chi-Square Test for Variance tests whether the variance of a sample differs
//! significantly from a specified population variance.
//!
//! # Functions
//!
//! - `goodness_of_fit`: Performs a Chi-Square Goodness of Fit Test.
//! - `independence`: Performs a Chi-Square Test for Independence.
//! - `variance`: Performs a Chi-Square Test for Variance.
//!
//! # Example
//!
//! ```rust
//! use crate::chi_square::{goodness_of_fit, independence, variance};
//! ```
pub mod categorical;
pub mod variance;

pub use categorical::{goodness_of_fit, independence};
pub use variance::variance;
