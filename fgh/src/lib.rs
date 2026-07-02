use pyo3::prelude::*;

/// A simple hello to verify the Rust → Python pipeline works.
#[pyfunction]
fn hello() -> String {
    "Hello from FGH (Rust + PyO3)!".to_string()
}

/// Add two integers — sanity check for argument passing.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[pymodule]
fn _fgh_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
