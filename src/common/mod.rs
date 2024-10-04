pub mod utils;
pub mod types;
pub mod calc;

pub use utils::mean_null_hypothesis;
pub use types::{TestResult,TailType};
pub use calc::{calculate_p_value as calculate_p, 
    calculate_confidence_interval as calculate_ci, 
    calculate_chi2_confidence_interval as calculate_chi2_ci};