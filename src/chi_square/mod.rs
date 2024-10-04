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
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for Chi-Square tests, you can use the following functions:
//! - `chi2_sample_size_gof`: Calculates the required sample size for the Chi-Square Goodness of Fit Test.
//! - `chi2_sample_size_ind`: Calculates the required sample size for the Chi-Square Test for Independence.
//! - `chi2_sample_size_variance`: Calculates the required sample size for the Chi-Square Test for Variance.
//!
//! # Functions
//!
//! - `goodness_of_fit`: Performs a Chi-Square Goodness of Fit Test.
//! - `independence`: Performs a Chi-Square Test for Independence.
//! - `variance`: Performs a Chi-Square Test for Variance.
//!
//! ## Example
//!
//! ```rust
//! use crate::chi_square::{goodness_of_fit, independence, variance, chi2_sample_size_gof, chi2_sample_size_ind, chi2_sample_size_variance};
//! ```
//!
//! ## Notes
//! Make sure to handle categorical data appropriately and verify assumptions before running the tests.

pub mod categorical;
pub mod sample_size;
pub mod variance;

pub use categorical::{goodness_of_fit, independence};
pub use sample_size::{chi2_sample_size_gof, chi2_sample_size_ind, chi2_sample_size_variance};
pub use variance::variance;
