#[macro_export]
macro_rules! register_module {
    ($module:ident, { $(fn $func_name:ident),* $(,)* ; $(type $type_name:ident),* $(,)* }) => {
        #[cfg(feature = "python_bindings")]
        pub fn register(py: pyo3::Python, m: &pyo3::types::PyModule) -> PyResult<()> {
            $(
                m.add_function(pyo3::wrap_pyfunction!($func_name, py)?)?; // Register functions
            )*
            $(
                m.add_class::<$type_name>()?; // Register types
            )*
            Ok(())
        }
    };
}
