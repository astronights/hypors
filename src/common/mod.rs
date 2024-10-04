pub mod calc;
pub mod types;
pub mod utils;

pub use calc::{
    calculate_chi2_confidence_interval as calculate_chi2_ci,
    calculate_confidence_interval as calculate_ci, calculate_p_value as calculate_p,
};
pub use types::{TailType, TestResult};
pub use utils::mean_null_hypothesis;
