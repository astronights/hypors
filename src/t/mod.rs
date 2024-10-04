//! # Student T Tests
//!
//! This module provides statistical t-test functionalities.
//!
//! The `t` module encompasses both one-sample and two-sample t-tests,
//! enabling users to perform hypothesis testing to compare means from sample data.
//!
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for t-tests, you can use the following function:
//! - `t_sample_size`: Calculates the necessary sample size for one-sample t-tests based on effect size, alpha, power, and standard deviation.
//!
//! ## Modules
//!
//! - `one_sample`: Contains functions for performing one-sample t-tests.
//! - `two_sample`: Contains functions for performing paired and independent two-sample t-tests.
//!
//! ## Exports
//!
//! The following functions are made available for use:
//!
//! - `t_test`: Performs a one-sample t-test.
//! - `t_test_ind`: Performs an independent two-sample t-test.
//! - `t_test_paired`: Performs a paired two-sample t-test.
//! - `t_sample_size`: Calculates the required sample size for one-sample t-tests.
//!
pub mod one_sample;
pub mod sample_size;
pub mod two_sample;

pub use one_sample::t_test;
pub use sample_size::t_sample_size;
pub use two_sample::{t_test_ind, t_test_paired};
