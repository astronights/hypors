pub mod one_sample;
pub mod two_sample;

pub use one_sample::z_test;
pub use two_sample::{z_test_ind, z_test_paired};
