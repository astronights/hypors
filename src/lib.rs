//! # HypoRS: A Statistical Hypothesis Testing Library
//!
//! `hypors` is a Rust library designed for performing a variety of hypothesis tests, including t-tests, z-tests, proportion tests, ANOVA, Chi-square tests, and Mann-Whitney tests. This library utilizes the `polars` crate for data manipulation and the `statrs` crate for statistical distributions.
//!
//! ## Overview
//!
//! The library is organized into the following key modules:
//!
//! - [`common`] - Contains shared utilities and helper functions for statistical calculations, including confidence intervals and p-values.
//! - [`t`] - Implements various t-tests, including one-sample, two-sample paired, and two-sample independent t-tests.
//! - [`z`] - Implements z-tests for one-sample and two-sample scenarios, supporting both paired and independent tests.
//! - [`proportion`] - Implements tests for proportions, including one-sample and two-sample proportion tests.
//! - [`anova`] - Implements one-way ANOVA tests for comparing means across multiple groups.
//! - [`chi_square`] - Implements Chi-square tests for categorical data analysis.
//! - [`mann_whitney`] - Implements the Mann-Whitney U test for comparing two independent samples.
//!
//! ### Sample Size Calculations
//!
//! Each of the parametrized tests have the faculty to also calculate minimum sample sizes required based on the Alpha and Power values.
//!
//! ## Hypothesis Tests
//!
//! ### T-Tests
//! Example of performing a one-sample t-test:
//! ```rust
//! use hypors::{t::t_test, common::TailType};
//!
//! let data = [1.2, 2.3, 1.9, 2.5, 2.8];
//! let population_mean = 2.0;
//! let tail = TailType::Two;
//! let alpha = 0.05;
//!
//! let result = t_test(data, population_mean, tail, alpha).unwrap();
//! println!("Test Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Confidence Interval: {:?}", result.confidence_interval);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! #### Features
//! - **One-sample t-test**: Tests whether the mean of a single sample differs from a specified population mean.
//! - **Two-sample paired t-test**: Tests whether the means of two related samples differ.
//! - **Two-sample independent t-test**: Tests whether the means of two unrelated samples differ, supporting both pooled and unpooled variances (Welch's t-test).
//! - **Sample Size Calculation**: Use `t_sample_size` to determine the required sample size for specified power and significance levels.
//!
//! ---
//!
//! ### Z-Tests
//! Example of performing a one-sample z-test:
//! ```rust
//! use hypors::{z::z_test, common::TailType};
//!
//! let data = vec![1.5, 2.3, 2.7, 2.8, 3.1];
//! let population_mean = 2.0;
//! let population_std_dev = 0.5;
//! let tail = TailType::Two;
//! let alpha = 0.05;
//!
//! let result = z_test(data, population_mean, population_std_dev, tail, alpha).unwrap();
//! println!("Z Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Confidence Interval: {:?}", result.confidence_interval);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! #### Features
//! - **One-sample z-test**: Tests whether the mean of a single sample differs from a specified population mean when the population standard deviation is known.
//! - **Two-sample paired z-test**: Tests whether the means of two related samples differ when the population standard deviation of the differences is known.
//! - **Two-sample independent z-test**: Tests whether the means of two unrelated samples differ, with options for pooled or unpooled variances, assuming known population standard deviations.
//! - **Sample Size Calculation**: Use `z_sample_size` to determine the required sample size for specified power and significance levels.
//!
//! ---
//!
//! ### Proportion Tests
//! Example of performing a one-sample proportion test:
//! ```rust
//! use hypors::{proportion::z_test, common::TailType};
//!
//! let successes = vec![1, 1, 0, 1, 0]; // Number of successes
//! let population_proportion = 0.25; // Population proportion
//! let tail = TailType::Right;
//! let alpha = 0.05;
//!
//! let result = z_test(successes, population_proportion, tail, alpha).unwrap();
//! println!("Test Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Confidence Interval: {:?}", result.confidence_interval);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! #### Features
//! - **One-sample proportion test**: Tests whether the proportion of successes in a single sample differs from a specified population proportion.
//! - **Two-sample proportion test**: Tests whether the proportions of successes in two independent samples differ.
//! - **Sample Size Calculation**: Use `prop_sample_size` to determine the required sample size for specified power and significance levels.
//!
//! ---
//!
//! ### ANOVA
//! Example of performing a one-way ANOVA test:
//! ```rust
//! use hypors::anova::anova;
//!
//! let group1 = vec![1.5, 2.5, 1.8];
//! let group2 = vec![2.3, 2.9, 3.0];
//! let group3 = vec![1.9, 2.2, 2.5];
//! let alpha = 0.05;
//!
//! let result = anova(&[group1, group2, group3], alpha).unwrap();
//! println!("F Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! #### Features
//! - **One-way ANOVA**: Tests whether at least one group mean differs from the others across multiple groups.
//! - **Sample Size Calculation**: Use `f_sample_size` to determine the required sample size for specified power and significance levels.
//!
//! ---
//!
//! ### Chi-Square Tests
//! Example of performing a Chi-square test for Goodness of Fit:
//! ```rust
//! use hypors::chi_square::goodness_of_fit;
//!
//! let observed = vec![10, 20, 30]; // Observed frequencies
//! let expected = vec![15, 15, 30]; // Expected frequencies
//! let alpha = 0.05;
//!
//! let result = goodness_of_fit(observed, expected, alpha).unwrap();
//! println!("Chi-Square Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! #### Features
//! - **Chi-square variance test**: Tests whether the variance of the distribution differs from the expected variance.
//! - **Chi-square test for independence**: Tests whether two categorical variables are independent of each other.
//! - **Chi-square goodness-of-fit test**: Tests whether the observed frequency distribution differs from the expected distribution.
//! - **Sample Size Calculation**: Use `chi2_sample_size_gof`, `chi2_sample_size_ind`,`chi2_sample_size_variance` to determine the required sample sizes for the different implementations respectively.
//!
//! ---
//!
//! ### Mann-Whitney U Test
//! Example of performing the Mann-Whitney U test:
//! ```rust
//! use hypors::mann_whitney::u_test;
//! use hypors::common::TailType;
//!
//! let group1 = vec![1.2, 2.3, 3.1];
//! let group2 = vec![2.5, 3.0, 3.8];
//! let alpha = 0.05;
//!
//! let result = u_test(group1, group2, alpha, TailType::Two).unwrap();
//! println!("U Statistic: {}", result.test_statistic);
//! println!("P-value: {}", result.p_value);
//! println!("Reject Null Hypothesis: {}", result.reject_null);
//! ```
//!
//! ####  Features
//! - **Mann-Whitney U test**: A non-parametric test used to determine whether there is a difference between two independent samples. This test is particularly useful when the data does not follow a normal distribution.
//!
//! ---
//!
//! ## Common Features
//!
//! - **Customizable tail type**: Supports left-tailed, right-tailed, and two-tailed tests for both t-tests and z-tests.
//! - **Confidence interval calculation**: Returns confidence intervals for all tests.
//!
//! ## Usage with Polars
//!
//! The library is designed to work with Vectors, arrays, and iterators of numeric data types.
//! It can also be integrated with the `polars` crate for more complex data manipulation tasks.
//! However, due to frequent minor version updates in `polars`, data should be converted to a
//! compatible format (e.g., `Vec<f64>`) before and after usage.
//!
//!
//! ## Crate Dependencies
//!
//! This library relies on the following crates:
//!
//! - [`statrs`](https://crates.io/crates/statrs) for statistical distributions.
//! - [`serde`](https://crates.io/crates/serde) for object serialization and deserialization.
//!
//! ## Error Handling
//!
//! The library uses `PolarsError` to handle errors that arise during data manipulation (e.g., failure to compute mean or variance). Each test function returns a `Result<TestResult, PolarsError>` type, where `TestResult` encapsulates the outcome of the hypothesis test.
//!
//! ## License
//!
//! This project is licensed under the MIT License.

pub mod common;

pub mod anova;
pub mod chi_square;
pub mod mann_whitney;
pub mod proportion;
pub mod t;
pub mod z;
