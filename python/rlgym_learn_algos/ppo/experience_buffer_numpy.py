# pyright: reportUnknownMemberType=false, reportIncompatibleVariableOverride=false, reportIncompatibleMethodOverride=false, reportMissingSuperCall=false

import os
import pickle
import zipfile
from collections.abc import Generator, Sequence
from io import BytesIO
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from rlgym.api import AgentID, RewardType
from typing_extensions import override

from ..util.torch_wrappers import TensorCircularBuffer
from .experience_buffer import EXPERIENCE_BUFFER_FILE, ExperienceBuffer
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
        self.log_probs: TensorCircularBuffer
        self.values: TensorCircularBuffer
        self.advantages: TensorCircularBuffer

    @override
    def _load_from_checkpoint(self):
        assert self.config.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint if checkpoint load folder is None!"
        )
        try:
            with zipfile.ZipFile(
                os.path.join(
                    self.config.checkpoint_load_folder, EXPERIENCE_BUFFER_FILE
                ),
                "r",
            ) as z:
                self.agent_ids = self._load_list_from_pkl(z, "agent_ids.pkl")
                self.observations = self._load_numpy_from_npy(z, "observations.npy")
                self.actions = self._load_numpy_from_npy(z, "actions.npy")
                self.log_probs = self._load_tensor_buffer_from_zip(z, "log_probs.pt")
                self.values = self._load_tensor_buffer_from_zip(z, "values.pt")
                self.advantages = self._load_tensor_buffer_from_zip(z, "advantages.pt")
        except FileNotFoundError:
            print(
                f"{self.config.agent_controller_name}: Tried to load experience buffer from checkpoint using the file at location {os.path.join(self.config.checkpoint_load_folder, EXPERIENCE_BUFFER_FILE)}, but there is no such file! A blank experience buffer will be used instead."
            )

    def _load_numpy_from_npy(self, z: zipfile.ZipFile, filename: str) -> np.ndarray:
        loaded_data = np.load(BytesIO(z.read(filename)), allow_pickle=False)
        loaded_len = len(loaded_data)
        if loaded_len > self.config.experience_buffer_config.max_size:
            print(
                f"{self.config.agent_controller_name}: Experience buffer checkpoint length for {filename} was {loaded_len}, but the configured capacity is {self.config.experience_buffer_config.max_size}. The newest samples that fit will be retained."
            )
            ret_arr = loaded_data[-self.config.experience_buffer_config.max_size :]
        else:
            ret_arr = loaded_data
        return ret_arr

    @staticmethod
    def _save_numpy_to_zip(z: zipfile.ZipFile, filename: str, v: np.ndarray):
        buf = BytesIO()
        np.save(buf, v, allow_pickle=False)
        z.writestr(filename, buf.getvalue())

    @override
    def save_checkpoint(self, folder_path: str | os.PathLike[str]):
        os.makedirs(folder_path, exist_ok=True)
        if self.config.experience_buffer_config.save_experience_buffer_in_checkpoint:
            with zipfile.ZipFile(
                os.path.join(folder_path, EXPERIENCE_BUFFER_FILE),
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as z:
                z.writestr("agent_ids.pkl", pickle.dumps(self.agent_ids))
                NumpyExperienceBuffer._save_numpy_to_zip(
                    z, "observations.npy", cast(np.ndarray, self.observations)
                )
                NumpyExperienceBuffer._save_numpy_to_zip(
                    z, "actions.npy", cast(np.ndarray, self.actions)
                )
                ExperienceBuffer._save_tensor_buffer_to_zip(
                    z, "log_probs.pt", self.log_probs
                )
                ExperienceBuffer._save_tensor_buffer_to_zip(z, "values.pt", self.values)
                ExperienceBuffer._save_tensor_buffer_to_zip(
                    z, "advantages.pt", self.advantages
                )
        self.trajectory_processor.save_checkpoint(folder_path)

    @override
    def submit_experience(
        self,
        trajectories: list[Trajectory[AgentID, np.ndarray, np.ndarray, RewardType]],
    ) -> TrajectoryProcessorData:
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
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.advantages.append(advantages)

        return trajectory_processor_data

    @override
    def _get_samples(
        self, indices: NDArray[np.int64]
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
            self.log_probs.tensor()[indices],
            self.values.tensor()[indices],
            self.advantages.tensor()[indices],
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
        total_samples = self.values.tensor().shape[0]
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
        del self.agent_ids
        del self.observations
        del self.actions
        del self.log_probs
        del self.values
        del self.advantages
        self.agent_ids = []
        self.observations = None
        self.actions = None
        self.log_probs = TensorCircularBuffer(
            capacity=self.max_size,
            shape=(),
            dtype=self.config.dtype,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.values = TensorCircularBuffer(
            capacity=self.max_size,
            shape=(),
            dtype=self.config.dtype,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.advantages = TensorCircularBuffer(
            capacity=self.max_size,
            shape=(),
            dtype=self.config.dtype,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
