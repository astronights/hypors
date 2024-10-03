//! # HypoRS: A Statistical Hypothesis Testing Library
//!
//! `hypors` is a Rust library designed to perform various hypothesis tests, including t-tests and other statistical analyses.
//! This library uses the `polars` crate for data manipulation and `statrs` for statistical distributions.
//!
//! ## Overview
//!
//! The library is divided into two main modules:
//!
//! - [`common`] - Contains common utilities, types, and helper functions used throughout the library, such as confidence interval calculations, p-value computation, and tail types.
//! - [`t_test`] - Implements t-tests, including one-sample, two-sample paired, and two-sample independent t-tests, with options for tail-type and pooled or unpooled variances.
//!
//! ## Example Usage
//!
//! Below is an example showing how to use the library to perform a one-sample t-test:
//!
//! ```rust
//! use polars::prelude::*;
//! use hypors::{t_test::one_sample, common::TailType};
//!
//! let data = Series::new("sample", &[1.2, 2.3, 1.9, 2.5, 2.8]);
//! let population_mean = 2.0;
//! let tail = TailType::Two;
//! let alpha = 0.05;
//!
//! let result = one_sample(&data, population_mean, tail, alpha).unwrap();
//!
//! println!("Test Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Confidence Interval: {:?}", result.confidence_interval);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! ## Modules
//!
//! - [`common`] - Defines shared utilities and types used in hypothesis testing, such as confidence interval and p-value calculations.
//! - [`t_test`] - Implements various t-test methods (one-sample, two-sample paired, and two-sample independent t-tests).
//!
//! ## Features
//!
//! - **One-sample t-test**: Test whether the mean of a single sample differs from a specified population mean.
//! - **Two-sample paired t-test**: Test whether the means of two related samples differ.
//! - **Two-sample independent t-test**: Test whether the means of two unrelated samples differ, with options for pooled or unpooled variances (Welch's t-test).
//! - **Customizable tail type**: Supports left-tailed, right-tailed, and two-tailed tests.
//! - **Confidence interval calculation**: Returns confidence intervals for all tests.
//!
//! ## Crate Dependencies
//!
//! This library relies on the following crates:
//!
//! - [`polars`](https://crates.io/crates/polars) for data manipulation and series handling.
//! - [`statrs`](https://crates.io/crates/statrs) for statistical distributions.
//!
//! ## Error Handling
//!
//! The library uses `PolarsError` to handle errors that arise during data manipulation (e.g., failure to compute mean or variance). Each test function returns a `Result<TestResult, PolarsError>` type, where `TestResult` encapsulates the outcome of the hypothesis test.
//!
//! ## License
//!
//! This project is licensed under the MIT License.

pub mod common;
pub mod t_test;
