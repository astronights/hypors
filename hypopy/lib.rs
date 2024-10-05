use pyo3::prelude::*;
use pyo3::types::PyModule;

// Modules
pub mod common;
pub mod python_bindings_macro;

#[pymodule]
fn hypopy(py: Python, m: &PyModule) -> PyResult<()> {
    let common_module = PyModule::new(py, "common")?; // Correctly create the common module
    common::register(py, common_module)?; // Register common functions and types
    m.add_submodule(common_module)?; // Correctly add common module to hypopy
    
    Ok(())
}
