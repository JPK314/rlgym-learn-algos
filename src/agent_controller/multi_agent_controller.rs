use std::collections::HashMap;

use itertools::{izip, Itertools};
use pyo3::exceptions::PyAssertionError;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};

use crate::agent_controller::EnvActionResponse;

#[allow(clippy::type_complexity)]
fn get_actions<'py>(
    agent_subcontroller: &Bound<'py, PyAny>,
    env_obs_data_dict: &HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
) -> PyResult<HashMap<u128, Bound<'py, PyAny>>> {
    agent_subcontroller
        .call_method1(
            intern!(agent_subcontroller.py(), "get_actions"),
            (env_obs_data_dict,),
        )?
        .extract()
}

fn choose_subcontrollers<'py>(
    py: Python<'py>,
    py_multi_agent_controller: &Bound<'py, PyAny>,
    env_agent_id_dict: &HashMap<u128, &Vec<Bound<'py, PyAny>>>,
) -> PyResult<HashMap<u128, Vec<String>>> {
    py_multi_agent_controller
        .call_method1(intern!(py, "choose_subcontrollers"), (env_agent_id_dict,))?
        .extract()
}

#[allow(clippy::type_complexity)]
fn choose_env_actions<'py>(
    py: Python<'py>,
    py_multi_agent_controller: &Bound<'py, PyAny>,
    state_info: &HashMap<u128, Bound<'py, PyAny>>,
) -> PyResult<(Py<PyAny>, HashMap<u128, Bound<'py, PyAny>>)> {
    py_multi_agent_controller
        .call_method1(intern!(py, "choose_env_actions"), (state_info,))?
        .extract()
}

fn step_env_action<'py>(
    py: Python<'py>,
    action_list: Vec<Bound<'py, PyAny>>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
) -> PyResult<Bound<'py, PyAny>> {
    static STEP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    STEP.get_or_try_init::<_, PyErr>(py, || {
        Ok(py
            .import("rlgym_learn")?
            .getattr("EnvAction")?
            .getattr("STEP")?
            .unbind())
    })?
    .bind(py)
    .call1((action_list, shared_info_setter_option, send_state))
}

fn reset_env_action<'py>(
    py: Python<'py>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
) -> PyResult<Bound<'py, PyAny>> {
    static RESET: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    RESET
        .get_or_try_init::<_, PyErr>(py, || {
            Ok(py
                .import("rlgym_learn")?
                .getattr("EnvAction")?
                .getattr("RESET")?
                .unbind())
        })?
        .bind(py)
        .call1((shared_info_setter_option, send_state))
}

fn set_state_env_action<'py>(
    py: Python<'py>,
    desired_state: Py<PyAny>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
    prev_timestep_id_dict_option: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    static SET_STATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    SET_STATE
        .get_or_try_init::<_, PyErr>(py, || {
            Ok(py
                .import("rlgym_learn")?
                .getattr("EnvAction")?
                .getattr("SET_STATE")?
                .unbind())
        })?
        .bind(py)
        .call1((
            desired_state,
            shared_info_setter_option,
            send_state,
            prev_timestep_id_dict_option,
        ))
}

#[pyclass(generic, module = "rlgym_learn._rlgym_learn")]
pub struct MultiAgentController {
    py_multi_agent_controller: Py<PyAny>,
    agent_subcontrollers: HashMap<String, Py<PyAny>>,
}

impl MultiAgentController {
    #[allow(clippy::type_complexity)]
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
        py_multi_agent_controller: &Bound<'py, PyAny>,
        env_obs_data_dict: HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
    ) -> PyResult<HashMap<u128, Vec<Bound<'py, PyAny>>>> {
        let n_envs = env_obs_data_dict.len();
        let mut agent_subcontrollers_env_actions_dict = env_obs_data_dict
            .iter()
            .map(|(&k, v)| (k, vec![None; v.0.len()]))
            .collect::<HashMap<_, _>>();
        let mut subcontroller_choices = choose_subcontrollers(
            py,
            py_multi_agent_controller,
            &env_obs_data_dict
                .iter()
                .map(|(&k, v)| (k, &v.0))
                .collect::<HashMap<u128, &Vec<Bound<'py, PyAny>>>>(),
        )?;

        let mut subcontrollers_env_obs_idx_dict = self
            .agent_subcontrollers
            .keys()
            .map(|name| (name, HashMap::with_capacity(n_envs)))
            .collect::<HashMap<_, _>>();

        for (env_id, (agent_id_list, obs_list)) in env_obs_data_dict.into_iter() {
            let len = obs_list.len();
            let env_subcontrollers = subcontroller_choices.remove(&env_id).ok_or_else(|| PyAssertionError::new_err("Returned dict from choose_subcontrollers did not contain keys for some env ids present in env_obs_data_dict"))?;
            if len != env_subcontrollers.len() {
                return Err(PyAssertionError::new_err("Returned dict from choose_subcontrollers does not contain subcontroller choices for all AgentID/ObsType pairs in env_obs_data_dict in all environments"));
            }
            let idx_list = (0..len).collect_vec();
            for (agent_id, obs, subcontroller, idx) in
                izip!(agent_id_list, obs_list, env_subcontrollers, idx_list)
            {
                let env_obs_idx = subcontrollers_env_obs_idx_dict
                    .get_mut(&subcontroller)
                    .ok_or_else(|| PyAssertionError::new_err("Returned dict from choose_subcontrollers contains subcontroller names not present in the agent_subcontrollers dict keys"))?
                    .entry(env_id)
                    .or_insert_with(|| {
                        (
                            Vec::with_capacity(len),
                            Vec::with_capacity(len),
                            Vec::with_capacity(len),
                        )
                    });
                env_obs_idx.0.push(agent_id);
                env_obs_idx.1.push(obs);
                env_obs_idx.2.push(idx);
            }
        }

        for (subcontroller, subcontroller_env_obs_idx_dict) in
            subcontrollers_env_obs_idx_dict.into_iter()
        {
            if subcontroller_env_obs_idx_dict.is_empty() {
                continue;
            }
            let (subcontroller_env_obs_data_dict, mut subcontroller_env_idx_dict) =
                subcontroller_env_obs_idx_dict
                    .into_iter()
                    .map(|(k, v)| ((k, (v.0, v.1)), (k, v.2)))
                    .collect::<(HashMap<_, _>, HashMap<_, _>)>();
            let subcontroller_env_actions_dict = get_actions(
                self.agent_subcontrollers
                    .get(subcontroller)
                    .unwrap()
                    .bind(py),
                &subcontroller_env_obs_data_dict,
            )?;
            for (env_id, batch_action) in subcontroller_env_actions_dict.into_iter() {
                let idx_list = subcontroller_env_idx_dict.remove(&env_id).unwrap();
                for (idx, action) in idx_list
                    .into_iter()
                    .zip(batch_action.extract::<Vec<Bound<'py, PyAny>>>()?)
                {
                    agent_subcontrollers_env_actions_dict
                        .get_mut(&env_id)
                        .unwrap()[idx] = Some(action);
                }
            }
        }

        Ok(agent_subcontrollers_env_actions_dict
            .into_iter()
            .map(|(env_id, action_list)| {
                (
                    env_id,
                    action_list
                        .into_iter()
                        .map(|action| action.unwrap())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>())
    }
}

#[pymethods]
impl MultiAgentController {
    #[new]
    pub fn new(
        py_multi_agent_controller: Py<PyAny>,
        agent_subcontrollers: HashMap<String, Py<PyAny>>,
    ) -> Self {
        MultiAgentController {
            py_multi_agent_controller,
            agent_subcontrollers,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn get_env_actions<'py>(
        &self,
        py: Python<'py>,
        mut env_obs_data_dict: HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
        env_state_info_dict: HashMap<u128, Bound<'py, PyAny>>,
    ) -> PyResult<(Py<PyAny>, Bound<'py, PyDict>)> {
        // Get env action responses from agent controllers
        let py_multi_agent_controller = self.py_multi_agent_controller.bind(py);
        let (n_new_envs, env_action_responses) =
            choose_env_actions(py, py_multi_agent_controller, &env_state_info_dict)?;

        if env_action_responses.len() != env_state_info_dict.len() {
            return Err(PyAssertionError::new_err("Returned dict from choose_env_actions does not contain env action choices for all environments included in env_state_info_dict"));
        }
        if !env_action_responses
            .keys()
            .all(|k| env_state_info_dict.contains_key(k))
        {
            return Err(PyAssertionError::new_err("Returned dict from choose_env_actions contains env action choices for env ids not present in env_state_info_dict"));
        }

        // Create env actions for RESET and SET_STATE variants
        let n_envs = env_obs_data_dict.len();
        let env_actions = PyDict::new(py);
        let mut step_env_action_responses = HashMap::with_capacity(n_envs);
        let mut should_get_actions = false;
        for (env_id, env_action_response) in env_action_responses.into_iter() {
            match env_action_response.extract::<EnvActionResponse>()? {
                EnvActionResponse::RESET {
                    shared_info_setter_option,
                    send_state,
                } => {
                    env_obs_data_dict.remove(&env_id);
                    env_actions.set_item(
                        env_id,
                        reset_env_action(py, shared_info_setter_option, send_state)?,
                    )?;
                }
                EnvActionResponse::SET_STATE {
                    desired_state,
                    shared_info_setter_option,
                    send_state,
                    prev_timestep_id_dict_option,
                } => {
                    env_obs_data_dict.remove(&env_id);
                    env_actions.set_item(
                        env_id,
                        set_state_env_action(
                            py,
                            desired_state,
                            shared_info_setter_option,
                            send_state,
                            prev_timestep_id_dict_option,
                        )?,
                    )?;
                }
                step_response => {
                    should_get_actions = true;
                    step_env_action_responses.insert(env_id, step_response);
                }
            };
        }

        // Handle getting actions for STEP variants
        if should_get_actions {
            let mut env_actions_dict =
                self.get_actions(py, py_multi_agent_controller, env_obs_data_dict)?;
            for (env_id, step_response) in step_env_action_responses.into_iter() {
                let EnvActionResponse::STEP {
                    shared_info_setter_option,
                    send_state,
                } = step_response
                else {
                    unreachable!();
                };
                let action_list = env_actions_dict.remove(&env_id).unwrap();
                env_actions.set_item(
                    env_id,
                    step_env_action(py, action_list, shared_info_setter_option, send_state)?,
                )?;
            }
        }

        Ok((n_new_envs, env_actions))
    }
}
