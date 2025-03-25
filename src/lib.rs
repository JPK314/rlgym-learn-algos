use pyo3::prelude::*;

pub mod common;
pub mod misc;
pub mod ppo;

#[pymodule]
#[pyo3(name = "rlgym_learn_algos")]
fn rlgym_learn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ppo::gae_trajectory_processor::GAETrajectoryProcessor>()?;
    m.add_class::<ppo::gae_trajectory_processor::DerivedGAETrajectoryProcessorConfig>()?;
    Ok(())
}
