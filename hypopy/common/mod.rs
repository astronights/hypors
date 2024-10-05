use crate::register_module;
use pyo3::{prelude::*, PyResult};

pub mod calc;
pub mod types;
pub mod utils;

pub use calc::{
    calculate_chi2_confidence_interval as calculate_chi2_ci,
    calculate_confidence_interval as calculate_ci,
    calculate_p_value as calculate_p,
};
pub use types::{TailType, TestResult};
pub use utils::mean_null_hypothesis;

// Register functions and types
register_module!(common, { 
    fn calculate_ci, 
    fn calculate_chi2_ci, 
    fn calculate_p; 
    type TailType, 
    type TestResult 
});
