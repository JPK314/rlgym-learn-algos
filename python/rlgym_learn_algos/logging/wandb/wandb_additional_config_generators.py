from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn.api import DerivedAgentControllerConfig

from ...ppo.ppo_agent_controller import PPOAgentControllerConfigModel
from ...ppo.trajectory_processor import TrajectoryProcessorConfig
from ..metrics_logger import MetricsLoggerConfig
from .wandb_metrics_logger import WandbAdditionalDerivedConfig


def ppo_additional_derived_config_factory(
    config: DerivedAgentControllerConfig[
        PPOAgentControllerConfigModel[TrajectoryProcessorConfig, MetricsLoggerConfig],
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    if (
        config.agent_controller_config.experience_buffer_config.trajectory_processor_config
        is not None
    ):
        trajectory_processor_fields = config.agent_controller_config.experience_buffer_config.trajectory_processor_config.model_dump()
    else:
        trajectory_processor_fields = {}
    return WandbAdditionalDerivedConfig(
        derived_wandb_run_config={
            **config.agent_controller_config.learner_config.model_dump(),
            "exp_buffer_size": config.agent_controller_config.experience_buffer_config.max_size,
            "timesteps_per_iteration": config.agent_controller_config.timesteps_per_iteration,
            "n_proc": config.process_config.n_proc,
            "min_process_steps_per_inference": config.process_config.min_process_steps_per_inference,
            "timestep_limit": config.base_config.timestep_limit,
            **trajectory_processor_fields,
        },
        run_suffix=config.agent_controller_config.run_suffix,
    )
