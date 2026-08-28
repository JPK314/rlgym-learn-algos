import json
import os
import random
from typing import cast

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from typing_extensions import override

from rlgym_learn_algos.stateful_functions import (
    BatchRewardTypeNumpyConverter,
    BatchRewardTypeSimpleNumpyConverter,
)
from rlgym_learn_algos.util.running_stats import WelfordRunningStat

from .gae_trajectory_processor import (
    GAETrajectoryProcessorConfigModel,
    GAETrajectoryProcessorData,
)
from .trajectory import Trajectory
from .trajectory_processor import (
    TRAJECTORY_PROCESSOR_FILE,
    DerivedTrajectoryProcessorConfig,
    TrajectoryProcessor,
)


class GAETrajectoryProcessorPurePython(
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
        self.return_running_stats: WelfordRunningStat = WelfordRunningStat((1,))
        self.config: (
            DerivedTrajectoryProcessorConfig[GAETrajectoryProcessorConfigModel] | None
        ) = None
        self.batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[
            RewardType
        ] = (
            batch_reward_type_numpy_converter
            if batch_reward_type_numpy_converter is not None
            else BatchRewardTypeSimpleNumpyConverter()
        )
        self.gamma: float
        self.lmbda: float
        self.standardize_rewards: bool
        self.max_returns_per_stats_increment: int | None
        self.dtype: np.dtype
        self.device: torch.device
        self.checkpoint_load_folder: str | None

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
            "process_trajectories cannot be called before load."
        )

        gamma = np.array(self.gamma, dtype=self.dtype)
        lmbda = np.array(self.lmbda, dtype=self.dtype)
        exp_len = 0
        agent_ids: list[AgentID] = []
        observations: list[ObsType] = []
        actions: list[ActionType] = []
        # For some reason, appending to lists is faster than preallocating the tensor and then indexing into it to assign
        log_probs_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []
        advantages_list: list[np.ndarray] = []
        returns_list: list[np.ndarray] = []
        reward_sum = np.array(0, dtype=self.dtype)

        trajectories_rewards_arrays = [
            self.batch_reward_type_numpy_converter.as_numpy(trajectory.reward_list)
            for trajectory in trajectories
        ]

        return_std = None
        if self.config.trajectory_processor_config.standardize_rewards:
            raw_returns_list: list[np.ndarray] = []
            for rewards_array in trajectories_rewards_arrays:
                raw_return = np.array(0, dtype=self.dtype)
                for reward in reversed(np.nditer(rewards_array)):
                    raw_return = reward + self.gamma * raw_return
                    raw_returns_list.append(raw_return)
            if (
                self.config.trajectory_processor_config.max_returns_per_stats_increment
                is not None
            ):
                for raw_return in random.sample(
                    raw_returns_list,
                    self.config.trajectory_processor_config.max_returns_per_stats_increment,
                ):
                    self.return_running_stats.update(raw_return)
            else:
                for raw_return in raw_returns_list:
                    self.return_running_stats.update(raw_return)
            return_std = self.return_running_stats.std.squeeze()

        for trajectory, rewards_array in zip(trajectories, trajectories_rewards_arrays):
            cur_return = np.array(0, dtype=self.dtype)
            next_val_pred = (
                cast(torch.Tensor, trajectory.final_val_pred).squeeze().cpu().numpy()
                if trajectory.truncated
                else np.array(0, dtype=self.dtype)
            )

            cur_advantages = np.array(0, dtype=self.dtype)
            value_preds = cast(torch.Tensor, trajectory.val_preds).unbind(0)
            for obs, action, log_prob, reward, value_pred in reversed(
                list(
                    zip(
                        trajectory.obs_list,
                        trajectory.action_list,
                        trajectory.log_probs,
                        np.nditer(rewards_array),
                        value_preds,
                    )
                )
            ):
                val_pred = value_pred.cpu().numpy()
                reward_sum += reward
                if return_std is not None:
                    norm_reward = reward / return_std
                else:
                    norm_reward = reward
                if self.config.trajectory_processor_config.reward_clip is not None:
                    norm_reward = np.clip(  # pyright: ignore [ reportUnknownMemberType]
                        norm_reward,
                        a_min=-self.config.trajectory_processor_config.reward_clip,
                        a_max=self.config.trajectory_processor_config.reward_clip,
                        dtype=self.dtype,
                    )
                delta = norm_reward + gamma * next_val_pred - val_pred
                next_val_pred = val_pred
                cur_advantages = delta + gamma * lmbda * cur_advantages
                cur_return = reward + gamma * cur_return
                returns_list.append(cur_return)
                agent_ids.append(trajectory.agent_id)
                observations.append(obs)
                actions.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value_pred)
                advantages_list.append(cur_advantages)
                exp_len += 1

        returns_array = np.array(returns_list)
        average_episode_return = reward_sum.item() / len(trajectories)
        avg_reward = reward_sum.item() / exp_len
        trajectory_processor_data = GAETrajectoryProcessorData(
            average_undiscounted_episodic_return=average_episode_return,
            average_return=returns_array.mean(),
            return_standard_deviation=returns_array.std(),
            average_reward=avg_reward,
        )
        return (
            (
                agent_ids,
                observations,
                actions,
                torch.stack(log_probs_list).to(device=self.device),
                torch.stack(values_list).to(device=self.device),
                torch.from_numpy(np.array(advantages_list)).to(device=self.device),  # pyright: ignore [reportUnknownMemberType]
            ),
            trajectory_processor_data,
        )

    @override
    def load(
        self,
        config: DerivedTrajectoryProcessorConfig[GAETrajectoryProcessorConfigModel],
    ):
        random.seed(config.seed)
        self.gamma = config.trajectory_processor_config.gamma
        self.lmbda = config.trajectory_processor_config.lmbda
        self.standardize_rewards = (
            config.trajectory_processor_config.standardize_rewards
        )
        self.max_returns_per_stats_increment = (
            config.trajectory_processor_config.max_returns_per_stats_increment
        )
        self.dtype = np.dtype(str(config.dtype).replace("torch.", ""))
        self.device = config.device
        self.checkpoint_load_folder = config.checkpoint_load_folder
        if self.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        self.batch_reward_type_numpy_converter.set_dtype(self.dtype)

    def _load_from_checkpoint(self):
        assert self.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint if checkpoint load folder is None!"
        )
        with open(
            os.path.join(self.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE),
            "rt",
        ) as f:
            state = json.load(f)
        self.return_running_stats.load_state_dict(state["return_running_stats"])

    @override
    def save_checkpoint(self, folder_path: str | os.PathLike[str]):
        state = {
            "return_running_stats": self.return_running_stats.state_dict(),
        }
        with open(
            os.path.join(folder_path, TRAJECTORY_PROCESSOR_FILE),
            "wt",
        ) as f:
            json.dump(state, f, indent=4)
