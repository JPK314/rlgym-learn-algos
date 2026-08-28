use core::slice;
use numpy::ndarray::Array1;
use numpy::ToPyArray;
use paste::paste;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use rand::prelude::*;

use super::trajectory::Trajectory;
use crate::common::NumpyDtype;
use crate::misc::torch_cat;
use crate::util::WelfordRunningStats;

#[derive(FromPyObject, Clone)]
pub struct GAETrajectoryProcessorConfig {
    gamma: Py<PyAny>,
    lmbda: Py<PyAny>,
    standardize_rewards: bool,
    max_returns_per_stats_increment: Option<usize>,
    reward_clip: Option<f64>,
}

fn extract_torch(torch_dtype: &Bound<'_, PyAny>) -> PyResult<NumpyDtype> {
    NumpyDtype::extract(
        torch_dtype
            .str()?
            .call_method1("replace", ("torch.", ""))?
            .as_borrowed(),
    )
}

#[derive(FromPyObject, Clone)]
pub struct DerivedGAETrajectoryProcessorConfig {
    trajectory_processor_config: GAETrajectoryProcessorConfig,
    agent_controller_name: String,
    #[pyo3(from_py_with = extract_torch)]
    dtype: NumpyDtype,
    seed: u64,
}

#[pyclass(module = "rlgym_learn_algos._rlgym_learn_algos", unsendable)]
pub struct GAETrajectoryProcessor {
    config: Option<DerivedGAETrajectoryProcessorConfig>,
    running_return_stats: Option<WelfordRunningStats>,
    rng: Option<SmallRng>,
    batch_reward_type_numpy_converter: Py<PyAny>,
}

#[pymethods]
impl GAETrajectoryProcessor {
    #[new]
    pub fn new(batch_reward_type_numpy_converter: Py<PyAny>) -> PyResult<Self> {
        Ok(GAETrajectoryProcessor {
            config: None,
            running_return_stats: None,
            rng: None,
            batch_reward_type_numpy_converter,
        })
    }

    pub fn load(&mut self, config: DerivedGAETrajectoryProcessorConfig) -> PyResult<()> {
        Python::attach(|py| {
            self.batch_reward_type_numpy_converter.call_method1(
                py,
                intern!(py, "set_dtype"),
                (config.dtype.into_pyobject(py)?,),
            )?;
            if config.trajectory_processor_config.standardize_rewards {
                self.running_return_stats = Some(WelfordRunningStats::default());
                self.rng = Some(SmallRng::seed_from_u64(config.seed));
            }
            self.config = Some(config);
            Ok(())
        })
    }

    pub fn load_state_dict<'py>(&mut self, state_dict: Option<Bound<'py, PyDict>>) -> PyResult<()> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("load_state_dict called before load"))?;
        if config.trajectory_processor_config.standardize_rewards {
            if let Some(state_dict) = state_dict {
                let mean = state_dict
                    .get_item("mean")?
                    .as_ref()
                    .map(PyAnyMethods::extract::<f64>);
                let count = state_dict
                    .get_item("count")?
                    .as_ref()
                    .map(PyAnyMethods::extract::<usize>);
                let m2 = state_dict
                    .get_item("m2")?
                    .as_ref()
                    .map(PyAnyMethods::extract::<f64>);

                match (mean, count, m2) {
                    (Some(Ok(mean)), Some(Ok(count)), Some(Ok(m2))) => {
                        self.running_return_stats = Some(WelfordRunningStats { mean, count, m2 });
                    }
                    _ => {
                        println!("{}: Warning: Running return stats state failed to be deserialized, using new running stats instead.", config.agent_controller_name);
                        self.running_return_stats = Some(WelfordRunningStats::default());
                    }
                }
            } else {
                self.running_return_stats = Some(WelfordRunningStats::default());
            }
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn process_trajectories<'py>(
        &mut self,
        py: Python<'py>,
        trajectories: Vec<Trajectory<'py>>,
    ) -> PyResult<(
        Vec<Bound<'py, PyAny>>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
    )> {
        let dtype = self
            .config
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("process_trajectories called before load"))?
            .dtype;
        match dtype {
            NumpyDtype::FLOAT32 => self.process_trajectories_f32(py, trajectories),

            NumpyDtype::FLOAT64 => self.process_trajectories_f64(py, trajectories),
            v => Err(PyNotImplementedError::new_err(format!(
                "GAE Trajectory Processor not implemented for dtype {:?}",
                v
            ))),
        }
    }

    pub fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(running_return_stats) = self.running_return_stats {
            let state_dict = PyDict::new(py);
            state_dict.set_item("mean", running_return_stats.mean)?;
            state_dict.set_item("count", running_return_stats.count)?;
            state_dict.set_item("m2", running_return_stats.m2)?;
            Ok(Some(state_dict))
        } else {
            Ok(None)
        }
    }
}

macro_rules! define_process_trajectories {
    (f32) => {
        static TORCH_F32_DTYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

        define_process_trajectories!(f32, |py| Ok::<_, PyErr>(
            TORCH_F32_DTYPE
            .get_or_try_init::<_, PyErr>(py, || {
                Ok(py
                    .import("torch")?
                    .getattr("float32")?
                    .unbind())
            })?
            .bind(py)
        ));
    };
    (f64) => {
        static TORCH_F64_DTYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

        define_process_trajectories!(f64, |py| Ok::<_, PyErr>(
            TORCH_F64_DTYPE
            .get_or_try_init::<_, PyErr>(py, || {
                Ok(py
                    .import("torch")?
                    .getattr("float64")?
                    .unbind())
            })?
            .bind(py)
        ));
    };
    ($dtype: ty, $torch_dtype_factory: expr) => {
        paste! {
            impl GAETrajectoryProcessor {
                fn [<process_trajectories_ $dtype>]<'py>(
                    &mut self,
                    py: Python<'py>,
                    trajectories: Vec<Trajectory<'py>>,
                ) -> PyResult<(
                    Vec<Bound<'py, PyAny>>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                    Bound<'py, PyAny>,
                )> {
                    let config = self.config.as_ref().unwrap();
                    let gamma = config.trajectory_processor_config.gamma.extract::<$dtype>(py)?;
                    let lambda = config.trajectory_processor_config.lmbda.extract::<$dtype>(py)?;
                    let batch_reward_type_numpy_converter = self.batch_reward_type_numpy_converter.bind(py);
                    let total_experience = trajectories
                        .iter()
                        .map(|trajectory| trajectory.obs_list.len())
                        .sum::<usize>();
                    let n_trajectories = trajectories.len();
                    let mut agent_id_list = Vec::with_capacity(total_experience);
                    let mut observation_list = Vec::with_capacity(total_experience);
                    let mut action_list = Vec::with_capacity(total_experience);
                    let mut log_probs_list = Vec::with_capacity(trajectories.len());
                    let mut values_list = Vec::with_capacity(trajectories.len());
                    let mut advantage_list = Vec::with_capacity(total_experience);
                    let mut return_list = Vec::with_capacity(total_experience);
                    let mut reward_sum = 0 as $dtype;
                    let trajectories_rewards_vecs = trajectories.iter().map(|trajectory| batch_reward_type_numpy_converter
                        .call_method1(intern!(py, "as_numpy"), (&trajectory.reward_list,))?
                        .extract::<Vec<$dtype>>()).collect::<PyResult<Vec<_>>>()?;

                    let mut return_std = 1 as $dtype;
                    if config.trajectory_processor_config.standardize_rewards {
                        let running_return_stats = self.running_return_stats.as_mut().ok_or_else(|| {
                            PyRuntimeError::new_err("Somehow running_return_stat got dropped")
                        })?;
                        let mut raw_returns = Vec::with_capacity(total_experience);
                        for rewards_vec in trajectories_rewards_vecs.iter() {
                            let mut raw_return = 0 as $dtype;
                            for reward in rewards_vec.iter().rev() {
                                raw_return = *reward + raw_return * gamma;
                                raw_returns.push(raw_return as f64);
                            }
                        }
                        if let Some(max_returns_per_stats_increment) = config.trajectory_processor_config.max_returns_per_stats_increment {
                            let (shuffled, _) = raw_returns.partial_shuffle(
                                self.rng
                                    .as_mut()
                                    .ok_or_else(|| PyRuntimeError::new_err("Somehow rng got dropped"))?,
                                max_returns_per_stats_increment,
                            );
                            shuffled
                                .iter()
                                .copied()
                                .for_each(|v| running_return_stats.update(v));
                        } else {
                            raw_returns
                                .into_iter()
                                .for_each(|v| running_return_stats.update(v));
                        }
                        return_std = running_return_stats.std_dev().unwrap_or(1_f64) as $dtype;
                        // Match the python running stats behavior. Seems a little weird to send a std of 0.01 to 0.01 but 0 to 1, but not really an issue
                        if 0 as $dtype == return_std {
                            return_std = 1 as $dtype;
                        }
                    }
                    let torch_dtype = ($torch_dtype_factory)(py)?;
                    for (trajectory, rewards_vec) in trajectories.into_iter().zip(trajectories_rewards_vecs.into_iter()) {
                        let trajectory_len = trajectory.obs_list.len();
                        let mut cur_return = 0 as $dtype;
                        let mut next_val_pred = if trajectory.truncated {
                            trajectory.final_val_pred.extract::<$dtype>()?
                        } else {
                            0 as $dtype
                        };
                        let mut cur_advantage = 0 as $dtype;
                        log_probs_list.push(trajectory.log_probs);
                        values_list.push(trajectory.val_preds.clone());
                        // TODO: add assertions for device, contiguity, dtype and check speed cost
                        let dtype = trajectory.val_preds.getattr(intern!(py, "dtype"))?;
                        assert!(dtype.eq(torch_dtype)?, "Value predictions must use dtype specified by config ({}), got {}", torch_dtype.repr()?.to_str()?, dtype.repr()?.to_str()?);
                        let device = trajectory.val_preds.getattr(intern!(py, "device"))?.getattr(intern!(py, "type"))?.extract::<String>()?;
                        assert!(device.eq("cpu"), "Value predictions must be on cpu, got {}", device);
                        assert!(trajectory.val_preds.call_method0(intern!(py, "is_contiguous"))?.extract::<bool>()?, "Value predictions must be a contiguous tensor");
                        let value_preds = unsafe {
                            let ptr = trajectory
                                .val_preds
                                .call_method0(intern!(py, "data_ptr"))?
                                .extract::<usize>()? as *const $dtype;
                            let mem = slice::from_raw_parts(
                                ptr,
                                trajectory
                                    .val_preds
                                    .call_method0(intern!(py, "numel"))?
                                    .extract::<usize>()?,
                            );
                            mem
                        };
                        let mut trajectory_agent_id_list = Vec::with_capacity(trajectory_len);
                        let mut trajectory_observation_list = Vec::with_capacity(trajectory_len);
                        let mut trajectory_action_list = Vec::with_capacity(trajectory_len);
                        let mut trajectory_advantage_list = Vec::with_capacity(trajectory_len);
                        let mut trajectory_return_list = Vec::with_capacity(trajectory_len);

                        for (obs, action, reward, &val_pred) in itertools::izip!(
                            trajectory.obs_list,
                            trajectory.action_list,
                            rewards_vec,
                            value_preds
                        ).rev()
                        {
                            reward_sum += reward;
                            let mut std_reward;
                            if config.trajectory_processor_config.standardize_rewards {
                                std_reward = reward / return_std;
                            } else {
                                std_reward = reward;
                            }
                            if let Some(reward_clip) = config.trajectory_processor_config.reward_clip {
                                std_reward = std_reward.min(reward_clip as $dtype).max(-reward_clip as $dtype);
                            }
                            let delta = std_reward + gamma * next_val_pred - val_pred;
                            next_val_pred = val_pred;
                            cur_advantage = delta + gamma * lambda * cur_advantage;
                            cur_return = reward + gamma * cur_return;
                            trajectory_agent_id_list.push(trajectory.agent_id.clone());
                            trajectory_observation_list.push(obs);
                            trajectory_action_list.push(action);
                            trajectory_advantage_list.push(cur_advantage);
                            trajectory_return_list.push(cur_return);
                        }
                        trajectory_agent_id_list.reverse();
                        trajectory_observation_list.reverse();
                        trajectory_action_list.reverse();
                        trajectory_advantage_list.reverse();
                        trajectory_return_list.reverse();
                        agent_id_list.append(&mut trajectory_agent_id_list);
                        observation_list.append(&mut trajectory_observation_list);
                        action_list.append(&mut trajectory_action_list);
                        advantage_list.append(&mut trajectory_advantage_list);
                        return_list.append(&mut trajectory_return_list);
                    }
                    Ok((
                        agent_id_list,
                        observation_list.into_pyobject(py)?,
                        action_list.into_pyobject(py)?,
                        torch_cat(py, &log_probs_list[..])?,
                        torch_cat(py, &values_list[..])?,
                        Array1::from_vec(advantage_list)
                            .to_pyarray(py)
                            .into_any(),
                        Array1::from_vec(return_list)
                            .to_pyarray(py)
                            .into_any(),
                        (reward_sum / (total_experience as $dtype)).into_pyobject(py)?.into_any(),
                        (reward_sum / (n_trajectories as $dtype)).into_pyobject(py)?.into_any()
                    ))
                }
            }
        }
    };
}

define_process_trajectories!(f64);
define_process_trajectories!(f32);
