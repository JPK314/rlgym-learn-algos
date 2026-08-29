from os import PathLike
from typing import Any, Generic

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from typing_extensions import override

from ..logging import DerivedMetricsLoggerConfig, DictMetricsLogger
from .gae_trajectory_processor import GAETrajectoryProcessorData
from .ppo_agent_controller import PPOAgentControllerConfigModel, PPOAgentControllerData
from .trajectory_processor import TrajectoryProcessorConfig


class PPOMetricsLogger(
    DictMetricsLogger[
        PPOAgentControllerConfigModel[TrajectoryProcessorConfig, Any],
        None,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        PPOAgentControllerData[GAETrajectoryProcessorData],
    ],
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    def __init__(self):
        self.state_metrics: dict[str, Any] = {}
        self.agent_metrics: dict[str, Any] = {}

    @property
    @override
    def config_model(self):
        return None

    @override
    def get_metrics(self) -> dict[str, Any]:
        return {**self.agent_metrics, **self.state_metrics}

    @override
    def collect_env_metrics(self, data: list[dict[str, Any] | None]):
        """
        Override this function to set self.state_metrics to something else using the data provided.
        The metrics should be nested dictionaries
        """
        self.state_metrics = {}

    @override
    def collect_agent_metrics(
        self, data: PPOAgentControllerData[GAETrajectoryProcessorData]
    ):
        self.agent_metrics = {
            "Timing": {
                "PPO Batch Consumption Time": data.ppo_data.batch_consumption_time,
                "Total Iteration Time": data.iteration_time,
                "Timestep Collection Time": data.timestep_collection_time,
                "Timestep Consumption Time": data.iteration_time
                - data.timestep_collection_time,
                "Collected Steps per Second": data.timesteps_collected
                / data.timestep_collection_time,
                "Overall Steps per Second": data.timesteps_collected
                / data.iteration_time,
            },
            "Timestep Collection": {
                "Cumulative Timesteps": data.cumulative_timesteps,
                "Timesteps Collected": data.timesteps_collected,
            },
            "PPO Metrics": {
                "Average Reward": data.trajectory_processor_data.average_reward,
                "Average Undiscounted Episodic Return": data.trajectory_processor_data.average_undiscounted_episodic_return,
                "Average Return": data.trajectory_processor_data.average_return,
                "Return Standard Deviation": data.trajectory_processor_data.return_standard_deviation,
                "Cumulative Model Updates": data.ppo_data.cumulative_model_updates,
                "Actor Entropy": data.ppo_data.actor_entropy,
                "Mean KL Divergence": data.ppo_data.kl_divergence,
                "Critic Loss": data.ppo_data.critic_loss,
                "SB3 Clip Fraction": data.ppo_data.sb3_clip_fraction,
                "Actor Update Magnitude": data.ppo_data.actor_update_magnitude,
                "Critic Update Magnitude": data.ppo_data.critic_update_magnitude,
            },
        }

    @override
    def load(
        self,
        config: DerivedMetricsLoggerConfig[
            PPOAgentControllerConfigModel[
                TrajectoryProcessorConfig,
                None,
            ],
            None,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ):
        pass

    @override
    def save_checkpoint(self, folder_path: str | PathLike[str]):
        pass
