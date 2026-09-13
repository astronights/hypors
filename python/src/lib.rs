//! Python bindings for the `hypors` hypothesis testing crate.
//!
//! Every statistic is computed by the parent crate; this layer only converts
//! Python objects to plain Rust values and back, so the two stay in step.

use pyo3::prelude::*;

mod anova;
mod chi_square;
mod convert;
mod mann_whitney;
mod proportion;
mod t;
mod types;
mod z;

/// Register `name` as a submodule of `parent`, and in `sys.modules` so that
/// `from hypors.t import t_test` works rather than only attribute access.
fn add_submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    // Fully qualified, so the module reports hypors.t rather than t.
    let module = PyModule::new(py, &format!("hypors.{name}"))?;
    register(&module)?;
    parent.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(format!("hypors.{name}"), &module)?;
    Ok(())
}

#[pymodule]
fn hypors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    add_submodule(m, "common", |m| {
        m.add_class::<types::TailType>()?;
        m.add_class::<types::TestResult>()?;
        Ok(())
    })?;
    add_submodule(m, "anova", anova::register)?;
    add_submodule(m, "chi_square", chi_square::register)?;
    add_submodule(m, "mann_whitney", mann_whitney::register)?;
    add_submodule(m, "proportion", proportion::register)?;
    add_submodule(m, "t", t::register)?;
    add_submodule(m, "z", z::register)?;

    // The two shared types are also available at the top level, since almost
    // every call needs them.
    m.add_class::<types::TailType>()?;
    m.add_class::<types::TestResult>()?;

    Ok(())
}
