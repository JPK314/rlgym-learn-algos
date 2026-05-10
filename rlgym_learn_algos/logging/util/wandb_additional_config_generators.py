from rlgym_learn.api import DerivedAgentControllerConfig

from ...ppo.ppo_agent_controller import PPOAgentControllerConfigModel
from ..wandb_metrics_logger import WandbAdditionalDerivedConfig


def generate_wandb_additional_derived_config_for_ppo(
    config: DerivedAgentControllerConfig[PPOAgentControllerConfigModel],
):
    return WandbAdditionalDerivedConfig(
        derived_wandb_run_config={
            **config.agent_controller_config.learner_config.model_dump(),
            "exp_buffer_size": config.agent_controller_config.experience_buffer_config.max_size,
            "timesteps_per_iteration": config.agent_controller_config.timesteps_per_iteration,
            "n_proc": config.process_config.n_proc,
            "min_process_steps_per_inference": config.process_config.min_process_steps_per_inference,
            "timestep_limit": config.base_config.timestep_limit,
            **config.agent_controller_config.experience_buffer_config.trajectory_processor_config,
        },
        run_suffix=config.agent_controller_config.run_suffix,
    )
