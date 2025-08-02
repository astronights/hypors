//! # Tests of Proportion
//!
//! The `proportion` module provides functions for performing proportion tests.
//!
//! Proportion tests are used to determine if there is a significant difference
//! between the proportions of two groups or to test a single proportion against a known value.
//!
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for proportion tests, you can use the following functions:
//! - `prop_sample_size`: Calculates the required sample size for proportion tests, including both one-sample and two-sample tests.
//!
//! ## Submodules
//!
//! - `one_sample`: Contains functions for conducting one-sample proportion tests.
//! - `two_sample`: Contains functions for conducting two-sample proportion tests.
//!
//! ## Exports
//!
//! The following functions are made available for use:
//!
//! - `z_test`: Performs a one-sample proportion test.
//! - `z_test_ind`: Performs a two-sample independent proportion test.
//!
//! ## Example
//! ```rust
//! use hypors::proportion::{z_test, z_test_ind, prop_sample_size};
//! ```

pub mod one_sample;
pub mod sample_size;
pub mod two_sample;

pub use one_sample::z_test;
pub use sample_size::prop_sample_size;
pub use two_sample::z_test_ind;
