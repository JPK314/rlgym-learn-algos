from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from numpy import ndarray
from pydantic import BaseModel
from rlgym.api import ActionType, AgentID, ObsType, RewardType

from rlgym_learn_algos.stateful_functions import (
    BatchRewardTypeNumpyConverter,
    BatchRewardTypeSimpleNumpyConverter,
)
from rlgym_learn_algos.util.running_stats import WelfordRunningStat

from ..ppo import RustDerivedGAETrajectoryProcessorConfig, RustGAETrajectoryProcessor
from .trajectory_processor import TRAJECTORY_PROCESSOR_FILE, TrajectoryProcessor


class GAETrajectoryProcessorConfigModel(BaseModel, extra="forbid"):
    gamma: float = 0.99
    lmbda: float = 0.95
    standardize_returns: bool = True
    max_returns_per_stats_increment: int = 150


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
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[
            RewardType
        ] = BatchRewardTypeSimpleNumpyConverter(),
    ):
        """
        :param batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter instance
        """
        self.return_stats = WelfordRunningStat(1)
        self.rust_gae_trajectory_processor: RustGAETrajectoryProcessor[
            AgentID, ObsType, ActionType, RewardType
        ] = RustGAETrajectoryProcessor(
            batch_reward_type_numpy_converter,
        )

    def process_trajectories(self, trajectories):
        return_std = self.return_stats.std[0] if self.standardize_returns else 1
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
        ) = self.rust_gae_trajectory_processor.process_trajectories(
            trajectories, return_std
        )

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(
                self.max_returns_per_stats_increment, len(return_array)
            )

            for sample in return_array[:n_to_increment]:
                self.return_stats.update(sample)
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
                log_probs.to(device=self.device),
                value_preds.to(device=self.device),
                torch.from_numpy(advantage_array).to(device=self.device),
            ),
            trajectory_processor_data,
        )

    def validate_config(self, config_obj):
        return GAETrajectoryProcessorConfigModel.model_validate(config_obj)

    def load(self, config):
        self.gamma = config.trajectory_processor_config.gamma
        self.lmbda = config.trajectory_processor_config.lmbda
        self.standardize_returns = (
            config.trajectory_processor_config.standardize_returns
        )
        self.max_returns_per_stats_increment = (
            config.trajectory_processor_config.max_returns_per_stats_increment
        )
        self.agent_controller_name = config.agent_controller_name
        self.dtype = config.dtype
        self.device = config.device
        self.checkpoint_load_folder = config.checkpoint_load_folder
        if self.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        self.rust_gae_trajectory_processor.load(
            RustDerivedGAETrajectoryProcessorConfig(
                self.gamma, self.lmbda, np.dtype(str(self.dtype)[6:])
            )
        )

    def _load_from_checkpoint(self):
        try:
            with open(
                os.path.join(self.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE),
                "rt",
            ) as f:
                state = json.load(f)
            self.return_stats.load_state_dict(state["return_running_stats"])
        except FileNotFoundError:
            print(
                f"{self.agent_controller_name}: Tried to load trajectory processor from checkpoint using the trajectory processor file at location {str(os.path.join(self.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE))}, but there is no such file! Running stats will be initialized as if this were a new run instead."
            )

    def save_checkpoint(self, folder_path):
        state = {
            "return_running_stats": self.return_stats.state_dict(),
        }
        with open(
            os.path.join(folder_path, TRAJECTORY_PROCESSOR_FILE),
            "wt",
        ) as f:
            json.dump(state, f, indent=4)
