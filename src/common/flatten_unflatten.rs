use std::collections::HashMap;

use itertools::izip;
use pyo3::{intern, prelude::*, types::PyDict};

#[pyclass]
pub struct FlattenedState {
    env_list: Vec<u128>,
    idx_list: Vec<usize>,
}

#[allow(clippy::type_complexity)]
#[pyfunction]
pub fn flatten_env_obs_data_dict<'py>(
    env_obs_data_dict: HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
) -> (
    (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>),
    FlattenedState,
) {
    let (agent_id_list, obs_list, env_list, idx_list) = env_obs_data_dict.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        |(mut acc_agent_id_list, mut acc_obs_list, mut acc_env_list, mut acc_idx_list),
         (env_id, (mut env_agent_id_list, mut env_obs_list))| {
            acc_agent_id_list.append(&mut env_agent_id_list);
            acc_obs_list.append(&mut env_obs_list);
            acc_env_list.push(env_id);
            acc_idx_list.push(acc_agent_id_list.len());
            (acc_agent_id_list, acc_obs_list, acc_env_list, acc_idx_list)
        },
    );
    (
        (agent_id_list, obs_list),
        FlattenedState { env_list, idx_list },
    )
}

#[pyfunction]
pub fn unflatten_iterable<'py>(
    py: Python<'py>,
    v: Vec<Bound<'py, PyAny>>,
    state: &FlattenedState,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut prev_idx = 0;
    for (&env_id, &idx) in izip!(state.env_list.iter(), state.idx_list.iter()) {
        d.set_item(env_id, &v[prev_idx..idx])?;
        prev_idx = idx;
    }
    Ok(d)
}

#[pyfunction]
pub fn unflatten_tensor<'py>(
    py: Python<'py>,
    t: Bound<'py, PyAny>,
    state: &FlattenedState,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut prev_idx = 0;
    for (&env_id, &idx) in izip!(state.env_list.iter(), state.idx_list.iter()) {
        d.set_item(
            env_id,
            t.call_method1(intern!(py, "narrow"), (0, prev_idx, idx - prev_idx))?,
        )?;
        prev_idx = idx;
    }
    Ok(d)
}
