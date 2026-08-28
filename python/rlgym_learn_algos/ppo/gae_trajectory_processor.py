from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from pydantic import BaseModel
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from typing_extensions import override

from .._rlgym_learn_algos.ppo import (
    GAETrajectoryProcessor as RustGAETrajectoryProcessor,
)
from ..stateful_functions import (
    BatchRewardTypeNumpyConverter,
    BatchRewardTypeSimpleNumpyConverter,
)
from .trajectory import Trajectory
from .trajectory_processor import (
    TRAJECTORY_PROCESSOR_FILE,
    DerivedTrajectoryProcessorConfig,
    TrajectoryProcessor,
)


class GAETrajectoryProcessorConfigModel(BaseModel, extra="forbid"):
    gamma: float = 0.99
    lmbda: float = 0.95
    standardize_rewards: bool = True
    max_returns_per_stats_increment: int | None = None
    reward_clip: float | None = 10


@dataclass
class GAETrajectoryProcessorData:
    average_reward: float
    average_undiscounted_episodic_return: float
    average_return: float
    return_standard_deviation: float


class GAETrajectoryProcessor(
    TrajectoryProcessor[
        GAETrajectoryProcessorConfigModel,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        GAETrajectoryProcessorData,
    ]
):
    def __init__(
        self,
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[RewardType]
        | None = None,
    ):
        """
        :param batch_reward_type_numpy_converter: Instance of BatchRewardTypeNumpyConverter to use.
        """
        self.rust_gae_trajectory_processor: RustGAETrajectoryProcessor[
            AgentID, ObsType, ActionType, RewardType
        ] = RustGAETrajectoryProcessor(
            batch_reward_type_numpy_converter
            if batch_reward_type_numpy_converter is not None
            else BatchRewardTypeSimpleNumpyConverter(),
        )
        self.config: (
            DerivedTrajectoryProcessorConfig[GAETrajectoryProcessorConfigModel] | None
        ) = None

    @property
    @override
    def config_model(self):
        return GAETrajectoryProcessorConfigModel

    @override
    def process_trajectories(
        self, trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]]
    ) -> tuple[
        tuple[
            list[AgentID],
            list[ObsType],
            list[ActionType],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        GAETrajectoryProcessorData,
    ]:
        assert self.config is not None, (
            "Cannot process trajectories before calling load with config!"
        )
        (
            agent_id_list,
            observation_list,
            action_list,
            log_probs,
            value_preds,
            advantage_array,
            return_array,
            avg_reward,
            avg_undiscounted_return,
        ) = self.rust_gae_trajectory_processor.process_trajectories(trajectories)

        trajectory_processor_data = GAETrajectoryProcessorData(
            average_reward=avg_reward,
            average_undiscounted_episodic_return=avg_undiscounted_return,
            average_return=return_array.mean(),
            return_standard_deviation=return_array.std(),
        )
        return (
            (
                agent_id_list,
                observation_list,
                action_list,
                log_probs.to(device=self.config.device),
                value_preds.to(device=self.config.device),
                torch.from_numpy(advantage_array).to(device=self.config.device),  # pyright: ignore [reportUnknownMemberType]
            ),
            trajectory_processor_data,
        )

    @override
    def load(
        self,
        config: DerivedTrajectoryProcessorConfig[GAETrajectoryProcessorConfigModel],
    ):
        self.config = config
        self.rust_gae_trajectory_processor.load(config)
        if config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

    # TODO: save/load running stats from rust side
    def _load_from_checkpoint(self):
        assert self.config is not None, (
            "Cannot load from checkpoint before calling load with config!"
        )
        assert self.config.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint if checkpoint load folder is None!"
        )
        try:
            with open(
                os.path.join(
                    self.config.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE
                ),
                "rt",
            ) as f:
                state = json.load(f)
            self.rust_gae_trajectory_processor.load_state_dict(
                state["return_running_stats"]
            )
        except FileNotFoundError:
            print(
                f"{self.config.agent_controller_name}: Tried to load trajectory processor from checkpoint using the trajectory processor file at location {os.path.join(self.config.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE)}, but there is no such file! Running stats will be initialized as if this were a new run instead."
            )

    @override
    def save_checkpoint(self, folder_path: str | os.PathLike[str]):
        state = {
            "return_running_stats": self.rust_gae_trajectory_processor.state_dict(),
        }
        with open(
            os.path.join(folder_path, TRAJECTORY_PROCESSOR_FILE),
            "wt",
        ) as f:
            json.dump(state, f, indent=4)
