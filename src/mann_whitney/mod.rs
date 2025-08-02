//! # Mann-Whitney U Tests
//!
//! The `mann_whitney` module provides functionality for performing the Mann-Whitney U Test.
//!
//! The Mann-Whitney U Test, also known as the Wilcoxon rank-sum test,
//! is a non-parametric test used to determine whether there is a significant
//! difference between the distributions of two independent groups.
//!
//! # Submodules
//!
//! - `u`: Contains the implementation of the Mann-Whitney U Test function.
//!
//! # Exports
//!
//! The following functions are made available for use:
//! - `u_test`: Performs the Mann-Whitney U Test for comparing two independent samples
//!
//! # Example
//! ```rust
//! use hypors::mann_whitney::u_test;
//! ```
pub mod u;

pub use u::u_test;
