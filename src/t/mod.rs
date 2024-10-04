pub mod one_sample;
pub mod two_sample;

pub use one_sample::t_test;
pub use two_sample::{t_test_ind, t_test_paired};