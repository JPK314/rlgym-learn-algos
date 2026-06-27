use pyo3::prelude::*;

mod common;
mod misc;
mod multi_agent_controller_manager;
mod ppo;

pub use common::{flatten_env_obs_data_dict, unflatten_iterable, unflatten_tensor};
use multi_agent_controller_manager::MultiAgentControllerManager;
pub use ppo::gae_trajectory_processor::{
    DerivedGAETrajectoryProcessorConfig, GAETrajectoryProcessor,
};

fn ppo<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "ppo")?;
    sub.add_class::<DerivedGAETrajectoryProcessorConfig>()?;
    sub.add_class::<GAETrajectoryProcessor>()?;
    parent.add_submodule(&sub)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rlgym_learn_algos._rlgym_learn_algos.ppo", &sub)?;

    Ok(())
}

fn util<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "util")?;
    sub.add_function(wrap_pyfunction!(flatten_env_obs_data_dict, py)?)?;
    sub.add_function(wrap_pyfunction!(unflatten_iterable, py)?)?;
    sub.add_function(wrap_pyfunction!(unflatten_tensor, py)?)?;
    parent.add_submodule(&sub)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rlgym_learn_algos._rlgym_learn_algos.util", &sub)?;

    Ok(())
}

fn agent_controller<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "agent_controller")?;
    sub.add_class::<MultiAgentControllerManager>()?;
    parent.add_submodule(&sub)?;
    py.import("sys")?.getattr("modules")?.set_item(
        "rlgym_learn_algos._rlgym_learn_algos.agent_controller",
        &sub,
    )?;

    Ok(())
}

#[pymodule]
mod _rlgym_learn_algos {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    #[pymodule_init]
    fn module_init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        ppo(py, m)?;
        util(py, m)?;
        agent_controller(py, m)?;
        Ok(())
    }
}
