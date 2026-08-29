use pyo3::{prelude::*, sync::PyOnceLock, types::PyAnyMethods, Python};

pub fn torch_cat<'py>(py: Python<'py>, obj: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
    static INTERNED_CAT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    INTERNED_CAT
        .get_or_try_init::<_, PyErr>(py, || Ok(py.import("torch")?.getattr("cat")?.unbind()))?
        .bind(py)
        .call1((obj,))
}
