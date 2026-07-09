use enum_kinds::EnumKind;
use pyo3::{
    prelude::*,
    types::{PyGenericAlias, PyType},
};

#[allow(non_camel_case_types)]
#[pyclass(from_py_object, module = "rlgym_learn_algos._rlgym_learn_algos")]
#[derive(Clone, Debug, EnumKind)]
#[enum_kind(
    EnvActionResponseType,
    allow(non_camel_case_types),
    pyclass(
        eq,
        eq_int,
        from_py_object,
        module = "rlgym_learn_algos._rlgym_learn_algos"
    )
)]
pub enum EnvActionResponse {
    #[pyo3(constructor = (shared_info_setter_option = None, send_state = false))]
    STEP {
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
    },
    #[pyo3(constructor = (shared_info_setter_option = None, send_state = false))]
    RESET {
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
    },
    #[pyo3(constructor = (desired_state, shared_info_setter_option = None, send_state = false, prev_timestep_id_dict_option = None))]
    SET_STATE {
        desired_state: Py<PyAny>,
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
        prev_timestep_id_dict_option: Option<Py<PyAny>>,
    },
}

#[pymethods]
impl EnvActionResponse {
    // python generics support
    #[classmethod]
    #[pyo3(signature = (key, /))]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyGenericAlias::new(cls.py(), cls.as_any(), key)?.into_any())
    }

    #[getter]
    fn enum_type(&self) -> EnvActionResponseType {
        match self {
            EnvActionResponse::STEP { .. } => EnvActionResponseType::STEP,
            EnvActionResponse::RESET { .. } => EnvActionResponseType::RESET,
            EnvActionResponse::SET_STATE { .. } => EnvActionResponseType::SET_STATE,
        }
    }

    #[getter]
    fn shared_info_setter<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        Ok(match self {
            EnvActionResponse::STEP {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option.as_ref().map(|v| v.clone_ref(py)),
            EnvActionResponse::RESET {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option.as_ref().map(|v| v.clone_ref(py)),
            EnvActionResponse::SET_STATE {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option.as_ref().map(|v| v.clone_ref(py)),
        })
    }

    #[getter]
    fn desired_state<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        if let EnvActionResponse::SET_STATE { desired_state, .. } = self {
            Ok(Some(desired_state.clone_ref(py)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn prev_timestep_id_dict<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        if let EnvActionResponse::SET_STATE {
            prev_timestep_id_dict_option,
            ..
        } = self
        {
            Ok(prev_timestep_id_dict_option
                .as_ref()
                .map(|v| v.clone_ref(py)))
        } else {
            Ok(None)
        }
    }
}
