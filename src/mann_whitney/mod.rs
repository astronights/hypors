//! # Mann-Whitney U Tests
//!
//! A module for performing the Mann-Whitney U Test.
//!
//! The Mann-Whitney U Test, also known as the Wilcoxon rank-sum test,
//! is a non-parametric test used to determine whether there is a significant 
//! difference between the distributions of two independent groups.
//!
//! # Submodules
//!
//! - `u`: Contains the implementation of the Mann-Whitney U Test function.
//!
//! # Usage
//!
//! To use the Mann-Whitney U Test, import the `u_test` function from this module:
//!
//! ```rust
//! use crate::mann_whitney::u_test;
//! ```
pub mod u;

pub use u::u_test;
