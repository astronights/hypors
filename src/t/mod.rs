//! # Student T Tests
//!
//! This module provides statistical t-test functionalities.
//!
//! The `t` module encompasses both one-sample and two-sample t-tests,
//! enabling users to perform hypothesis testing to compare means from sample data.
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
pub mod one_sample;
pub mod two_sample;

pub use one_sample::t_test;
pub use two_sample::{t_test_ind, t_test_paired};
