//! # Tests of Proportion
//!
//! The `proportion` module provides functions for performing proportion tests.
//!
//! This module includes both one-sample and two-sample proportion tests to assess
//! the significance of sample proportions in relation to hypothesized population proportions
//! or between two independent samples.
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
pub mod one_sample;
pub mod two_sample;

pub use one_sample::z_test;
pub use two_sample::z_test_ind;
