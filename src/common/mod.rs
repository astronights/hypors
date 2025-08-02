//! # Common Utilities
//!
//! This module provides various statistical functions and types used in hypothesis testing and confidence interval calculations.
//!
//! It is organized into submodules:
//!
//! - `calc`: Contains functions for calculating p-values, confidence intervals, and Chi-squared confidence intervals.
//! - `types`: Defines types such as `TailType` and `TestResult` used in statistical analysis.
//! - `utils`: Contains utility functions for hypothesis creation and related tasks.
//!
//! # Re-exports
//!
//! The following functions and types are re-exported for convenience:
//!
//! - `calculate_chi2_ci`: Alias for `calculate_chi2_confidence_interval` function from the `calc` module.
//! - `calculate_ci`: Alias for `calculate_confidence_interval` function from the `calc` module.
//! - `calculate_p`: Alias for `calculate_p_value` function from the `calc` module.
//! - `TailType`: The enumeration representing different types of tails in hypothesis testing from the `types` module.
//! - `TestResult`: The structure that holds the results of a statistical test from the `types` module.
//! - `mean_null_hypothesis`: A utility function for generating null hypothesis strings from the `utils` module.

pub mod calc;
pub mod errors;
pub mod types;
pub mod utils;

pub use calc::{
    calculate_chi2_confidence_interval as calculate_chi2_ci,
    calculate_confidence_interval as calculate_ci, calculate_p_value as calculate_p,
};
pub use errors::StatError;
pub use types::{TailType, TestResult};
pub use utils::mean_null_hypothesis;
