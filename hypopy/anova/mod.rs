use pyo3::prelude::*;

pub mod one_way;
pub mod sample_size;

pub use one_way::py_anova as anova;
pub use sample_size::py_f_sample_size as f_sample_size;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(anova, m)?)?;
    m.add_function(wrap_pyfunction!(f_sample_size, m)?)?;

    Ok(())
}