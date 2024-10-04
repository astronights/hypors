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
//! use polars::prelude::*;
//! use hypors::{t::one_sample, common::TailType};
//!
//! let data = Series::new("sample", &[1.2, 2.3, 1.9, 2.5, 2.8]);
//! let population_mean = 2.0;
//! let tail = TailType::Two;
//! let alpha = 0.05;
//!
//! let result = one_sample(&data, population_mean, tail, alpha).unwrap();
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
//! use polars::prelude::*;
//! use hypors::{z::one_sample_z, common::TailType};
//!
//! let data = Series::new("sample", &[1.5, 2.3, 2.7, 2.8, 3.1]);
//! let population_mean = 2.0;
//! let population_std_dev = 0.5;
//! let tail = TailType::Two;
//! let alpha = 0.05;
//!
//! let result = one_sample_z(&data, population_mean, population_std_dev, tail, alpha).unwrap();
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
//! use polars::prelude::*;
//! use hypors::{proportion::one_sample_proportion, common::TailType};
//!
//! let successes = 30; // Number of successes
//! let sample_size = 100; // Total sample size
//! let population_proportion = 0.25; // Population proportion
//! let tail = TailType::Right;
//! let alpha = 0.05;
//!
//! let result = one_sample_proportion(successes, sample_size, population_proportion, tail, alpha).unwrap();
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
//! use polars::prelude::*;
//! use hypors::anova::one_way_anova;
//!
//! let group1 = Series::new("Group 1", &[1.5, 2.5, 1.8]);
//! let group2 = Series::new("Group 2", &[2.3, 2.9, 3.0]);
//! let group3 = Series::new("Group 3", &[1.9, 2.2, 2.5]);
//!
//! let result = one_way_anova(&[group1, group2, group3]).unwrap();
//! println!("F Statistic: {}", result.f_statistic);
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
//! Example of performing a Chi-square test for independence:
//! ```rust
//! use polars::prelude::*;
//! use hypors::chi_square::chi_square_test;
//!
//! let observed = vec![10, 20, 30]; // Observed frequencies
//! let expected = vec![15, 15, 30]; // Expected frequencies
//!
//! let result = chi_square_test(&observed, &expected).unwrap();
//! println!("Chi-Square Statistic: {}", result.chi_square_statistic);
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
//! use polars::prelude::*;
//! use hypors::mann_whitney::mann_whitney_u;
//!
//! let group1 = Series::new("Group 1", &[1.2, 2.3, 3.1]);
//! let group2 = Series::new("Group 2", &[2.5, 3.0, 3.8]);
//!
//! let result = mann_whitney_u(&group1, &group2).unwrap();
//! println!("U Statistic: {}", result.u_statistic);
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
//! ## Crate Dependencies
//!
//! This library relies on the following crates:
//!
//! - [`polars`](https://crates.io/crates/polars) for data manipulation and series handling.
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
