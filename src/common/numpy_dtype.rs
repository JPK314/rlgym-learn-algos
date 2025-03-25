use numpy::{dtype, Element, PyArrayDescr, PyArrayDescrMethods};
use pyo3::{exceptions::PyValueError, intern, prelude::*};

// Why not just use PyArrayDescr? Because PyArrayDescr doesn't allow for derivation of Debug, PartialEq, or Copy.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NumpyDtype {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
}

impl<'py> IntoPyObject<'py> for NumpyDtype {
    type Target = PyArrayDescr;

    type Output = Bound<'py, PyArrayDescr>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            NumpyDtype::INT8 => dtype::<i8>(py),
            NumpyDtype::INT16 => dtype::<i16>(py),
            NumpyDtype::INT32 => dtype::<i32>(py),
            NumpyDtype::INT64 => dtype::<i64>(py),
            NumpyDtype::UINT8 => dtype::<u8>(py),
            NumpyDtype::UINT16 => dtype::<u16>(py),
            NumpyDtype::UINT32 => dtype::<u32>(py),
            NumpyDtype::UINT64 => dtype::<u64>(py),
            NumpyDtype::FLOAT32 => dtype::<f32>(py),
            NumpyDtype::FLOAT64 => dtype::<f64>(py),
        })
    }
}

impl<'py> FromPyObject<'py> for NumpyDtype {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let dtype_any = py
            .import(intern!(py, "numpy"))?
            .getattr(intern!(py, "dtype"))?
            .call1((ob,))?;
        let dtype = dtype_any.downcast::<PyArrayDescr>()?;
        if dtype.is_equiv_to(&i8::get_dtype(py)) {
            Ok(NumpyDtype::INT8)
        } else if dtype.is_equiv_to(&u8::get_dtype(py)) {
            Ok(NumpyDtype::UINT8)
        } else if dtype.is_equiv_to(&i16::get_dtype(py)) {
            Ok(NumpyDtype::INT16)
        } else if dtype.is_equiv_to(&u16::get_dtype(py)) {
            Ok(NumpyDtype::UINT16)
        } else if dtype.is_equiv_to(&i32::get_dtype(py)) {
            Ok(NumpyDtype::INT32)
        } else if dtype.is_equiv_to(&u32::get_dtype(py)) {
            Ok(NumpyDtype::UINT32)
        } else if dtype.is_equiv_to(&i64::get_dtype(py)) {
            Ok(NumpyDtype::INT64)
        } else if dtype.is_equiv_to(&u64::get_dtype(py)) {
            Ok(NumpyDtype::UINT64)
        } else if dtype.is_equiv_to(&f32::get_dtype(py)) {
            Ok(NumpyDtype::FLOAT32)
        } else if dtype.is_equiv_to(&f64::get_dtype(py)) {
            Ok(NumpyDtype::FLOAT64)
        } else {
            Err(PyValueError::new_err(format!(
                "Invalid dtype: {}",
                dtype.repr()?.to_str()?
            )))
        }
    }
}
