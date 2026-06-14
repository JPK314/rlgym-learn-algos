# pyright: reportUnknownMemberType=false, reportIncompatibleVariableOverride=false, reportIncompatibleMethodOverride=false, reportMissingSuperCall=false

from collections.abc import Generator, Sequence
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from rlgym.api import AgentID, RewardType
from typing_extensions import override

from .experience_buffer import ExperienceBuffer
from .trajectory import Trajectory
from .trajectory_processor import (
    TrajectoryProcessor,
    TrajectoryProcessorConfig,
    TrajectoryProcessorData,
)


class NumpyExperienceBuffer(
    ExperienceBuffer[
        TrajectoryProcessorConfig,
        AgentID,
        np.ndarray,
        np.ndarray,
        RewardType,
        TrajectoryProcessorData,
    ],
):
    @staticmethod
    def _cat_numpy(t1: np.ndarray | None, t2: np.ndarray, size: int):
        t2_len = len(t2)
        if t1 is None:
            if t2_len > size:
                t = cast(np.ndarray, t2[-size:].copy())
            else:
                t = t2
            del t1
            del t2
            return t

        if t2_len > size:
            # t2 alone is larger than we want; copy the end
            # This copy is needed to avoid nesting views
            t = cast(np.ndarray, t2[-size:].copy())

        elif t2_len == size:
            # t2 is a perfect match; just use it directly
            t = t2

        elif len(t1) + t2_len > size:
            # t1+t2 is larger than we want; use t2 wholly with the end of t1 before it
            t = np.concatenate((t1[t2_len - size :], t2), 0)

        else:
            # t1+t2 does not exceed what we want; concatenate directly
            t = np.concatenate((t1, t2), 0)

        del t1
        del t2
        return t

    def __init__(
        self,
        trajectory_processor: TrajectoryProcessor[
            TrajectoryProcessorConfig,
            AgentID,
            np.ndarray,
            np.ndarray,
            RewardType,
            TrajectoryProcessorData,
        ],
    ):
        self.trajectory_processor: TrajectoryProcessor[
            TrajectoryProcessorConfig,
            AgentID,
            np.ndarray,
            np.ndarray,
            RewardType,
            TrajectoryProcessorData,
        ] = trajectory_processor
        self.agent_ids: list[AgentID] = []
        self.observations: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.log_probs: torch.Tensor = torch.FloatTensor()
        self.values: torch.Tensor = torch.FloatTensor()
        self.advantages: torch.Tensor = torch.FloatTensor()

    @override
    def submit_experience(
        self,
        trajectories: list[Trajectory[AgentID, np.ndarray, np.ndarray, RewardType]],
    ) -> TrajectoryProcessorData:
        _cat = ExperienceBuffer._cat
        _cat_list = ExperienceBuffer._cat_list
        _cat_numpy = NumpyExperienceBuffer._cat_numpy
        exp_buffer_data, trajectory_processor_data = (
            self.trajectory_processor.process_trajectories(trajectories)
        )
        (agent_ids, observations, actions, log_probs, values, advantages) = (
            exp_buffer_data
        )

        self.agent_ids = _cat_list(
            self.agent_ids, agent_ids, self.config.experience_buffer_config.max_size
        )
        self.observations = _cat_numpy(
            self.observations,
            np.array(observations),
            self.config.experience_buffer_config.max_size,
        )
        self.actions = _cat_numpy(
            self.actions,
            np.array(actions),
            self.config.experience_buffer_config.max_size,
        )
        self.log_probs = _cat(
            self.log_probs,
            log_probs,
            self.config.experience_buffer_config.max_size,
        )
        self.values = _cat(
            self.values,
            values,
            self.config.experience_buffer_config.max_size,
        )
        self.advantages = _cat(
            self.advantages,
            advantages,
            self.config.experience_buffer_config.max_size,
        )

        return trajectory_processor_data

    @override
    def _get_samples(
        self, indices: NDArray[np.long]
    ) -> tuple[
        Sequence[AgentID],
        np.ndarray,
        np.ndarray,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.observations is not None and self.actions is not None, (
            "Can't get samples before any data has been added to the experience buffer!"
        )
        py_indices: list[int] = indices.tolist()
        return (
            [self.agent_ids[index] for index in py_indices],
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.values[indices],
            self.advantages[indices],
        )

    @override
    def get_all_batches_shuffled(
        self, batch_size: int
    ) -> Generator[
        tuple[
            Sequence[AgentID],
            np.ndarray,
            np.ndarray,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Any,
        None,
    ]:
        """
        Function to return the experience buffer in shuffled batches. Code taken from the stable-baeselines3 buffer:
        https://github.com/DLR-RM/stable-baselines3/blob/2ddf015cd9840a2a1675f5208be6eb2e86e4d045/stable_baselines3/common/buffers.py#L482
        :param batch_size: size of each batch yielded by the generator.
        :return:
        """
        if self.config.experience_buffer_config.device.type != "cpu":
            torch.cuda.current_stream().synchronize()
        total_samples = self.values.shape[0]
        indices = self.rng.permutation(total_samples)
        start_idx = 0
        while start_idx + batch_size <= total_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    @override
    def clear(self):
        """
        Function to clear the experience buffer.
        :return: None.
        """
        del self.observations
        del self.actions
        del self.log_probs
        del self.values
        del self.advantages
        self.agent_ids = []
        self.observations = None
        self.actions = None
        self.log_probs = torch.FloatTensor()
        self.values = torch.FloatTensor()
        self.advantages = torch.FloatTensor()
