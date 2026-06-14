use pyo3::prelude::*;

pub mod common;
pub mod misc;
pub mod ppo;

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

#[pymodule]
mod _rlgym_learn_algos {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    #[pymodule_init]
    fn module_init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        ppo(py, m)?;
        Ok(())
    }
}
