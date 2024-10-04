//! # Z Tests
//!
//! This module provides implementations for Z-tests.
//!
//! Z-tests are statistical tests used to determine if there is a significant difference
//! between sample means or proportions. They are applicable when the population standard
//! deviation is known and the sample size is large (typically n > 30).
//!
//! ## Sample Size Calculation
//!
//! To calculate the required sample size for Z-tests, you can use the following function:
//! - `z_sample_size`: Calculates the necessary sample size for one-sample and two-sample Z-tests based on desired power, significance level, and effect size.
//!
//! ## Module Overview
//!
//! The `z` module contains functions for performing the following types of Z-tests:
//!
//! - **One-sample Z-test**: Tests whether the mean of a single sample differs from a known population mean.
//! - **Two-sample Z-tests**:
//!     - **Paired Z-test**: Tests the means of two related samples.
//!     - **Independent Z-test**: Tests the means of two independent samples.
//!
//! ## Submodules
//!
//! - `one_sample`: Contains functionality for conducting one-sample Z-tests.
//! - `two_sample`: Contains functionality for conducting paired and independent two-sample Z-tests.
//!
//! ## Usage Example
//!
//! ```rust
//! use polars::prelude::*;
//! use hypors::z::{z_test, z_test_ind, TailType};
//!
//! // Example for a one-sample Z-test
//! let series = Series::new("data", &[10.0, 12.0, 14.0]);
//! let pop_mean = 11.0; // Known population mean
//! let pop_std = 2.0; // Known population standard deviation
//! let tail = TailType::Two; // Two-tailed test
//! let alpha = 0.05; // 5% significance level
//!
//! // Perform the one-sample Z-test
//! let result_one_sample = z_test(&series, pop_mean, pop_std, tail, alpha).unwrap();
//!
//! // Example for an independent Z-test
//! let series1 = Series::new("group1", &[5.0, 6.0, 7.0]);
//! let series2 = Series::new("group2", &[8.0, 9.0, 10.0]);
//! let pop_std1 = 1.0; // Population standard deviation for group 1
//! let pop_std2 = 1.0; // Population standard deviation for group 2
//!
//! // Perform the independent two-sample Z-test
//! let result_ind = z_test_ind(&series1, &series2, pop_std1, pop_std2, tail, alpha).unwrap();
//! ```
//!
//! ## Exports
//!
//! The following functions are made available for use:
//!
//! - `z_test`: Performs a one-sample Z-test.
//! - `z_test_ind`: Performs an independent two-sample Z-test.
//! - `z_test_paired`: Performs a paired two-sample Z-test.
//! - `z_sample_size`: Calculates the required sample size for one-sample and two-sample Z-tests.
//!
pub mod one_sample;
pub mod sample_size;
pub mod two_sample;

pub use one_sample::z_test;
pub use sample_size::z_sample_size;
pub use two_sample::{z_test_ind, z_test_paired};
