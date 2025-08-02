//! # Z Tests
//!
//! The `z` module provides functionality for performing Z-tests.
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
//! ## Submodules
//!
//! - `one_sample`: Contains functionality for conducting one-sample Z-tests.
//! - `two_sample`: Contains functionality for conducting paired and independent two-sample Z-tests.
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
//! ## Example
//! ```rust
//! use hypors::z::{z_test, z_test_ind, z_test_paired, z_sample_size};
//! ```

pub mod one_sample;
pub mod sample_size;
pub mod two_sample;

pub use one_sample::z_test;
pub use sample_size::z_sample_size;
pub use two_sample::{z_test_ind, z_test_paired};
