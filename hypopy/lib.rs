use pyo3::prelude::*;

pub mod common;
pub mod anova;

#[pymodule]
fn hypors(m: &Bound<'_, PyModule>) -> PyResult<()> {

    let common_module = PyModule::new_bound(m.py(), "common")?;
    common::register(&common_module)?;
    m.add_submodule(&common_module)?;

    let anova_module = PyModule::new_bound(m.py(), "anova")?;
    anova::register(&anova_module)?;
    m.add_submodule(&anova_module)?;

    m.py().import_bound("sys")?
        .getattr("modules")?
        .set_item("hypors.common", common_module)?;

    m.py().import_bound("sys")?
        .getattr("modules")?
        .set_item("hypors.anova", anova_module)?;
    
    Ok(())
}