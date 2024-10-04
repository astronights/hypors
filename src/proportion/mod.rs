//! # Tests of Proportion
//!
//! The `proportion` module provides functions for performing proportion tests.
//!
//! This module includes both one-sample and two-sample proportion tests to assess
//! the significance of sample proportions in relation to hypothesized population proportions
//! or between two independent samples.
//!
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for proportion tests, you can use the following functions:
//! - `prop_sample_size`: Calculates the required sample size for proportion tests, including both one-sample and two-sample tests.
//!
//! # Submodules
//!
//! - `one_sample`: Contains functions for conducting one-sample proportion tests.
//! - `two_sample`: Contains functions for conducting two-sample proportion tests.
//!
//! # Public Functions
//!
//! - `z_test`: Performs a one-sample proportion test.
//! - `z_test_ind`: Performs a two-sample independent proportion test.
//!
//! ## Example
//! ```rust
//! use crate::proportion::{z_test, z_test_ind, prop_sample_size};
//! ```

pub mod one_sample;
pub mod sample_size;
pub mod two_sample;

pub use one_sample::z_test;
pub use sample_size::prop_sample_size;
pub use two_sample::z_test_ind;
