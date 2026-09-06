# pyright: reportUnknownMemberType=false, reportIncompatibleVariableOverride=false, reportIncompatibleMethodOverride=false, reportMissingSuperCall=false

import os
import pickle
import zipfile
from collections.abc import Generator, Sequence
from io import BytesIO
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from rlgym.api import ActionSpaceType, AgentID, ObsSpaceType, RewardType
from typing_extensions import override

from ..util.circular_buffers import NumpyCircularBuffer, TensorCircularBuffer
from ..util.torch_utils import torch_to_numpy_dtype
from .experience_buffer import (
    EXPERIENCE_BUFFER_FILE,
    DerivedExperienceBufferConfig,
    ExperienceBuffer,
)
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
        ObsSpaceType,
        ActionSpaceType,
        TrajectoryProcessorData,
    ],
):
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
        self.observations: NumpyCircularBuffer
        self.actions: NumpyCircularBuffer
        self.log_probs: TensorCircularBuffer
        self.values: TensorCircularBuffer
        self.advantages: TensorCircularBuffer

    @override
    def load(
        self,
        config: DerivedExperienceBufferConfig[
            TrajectoryProcessorConfig, ObsSpaceType, ActionSpaceType
        ],
    ):
        self.observations = NumpyCircularBuffer(
            capacity=config.experience_buffer_config.max_size,
            forced_dtype=torch_to_numpy_dtype(config.dtype),
        )
        self.actions = NumpyCircularBuffer(
            capacity=config.experience_buffer_config.max_size, forced_dtype=None
        )
        super().load(config)

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
                self.agent_ids = self._load_list_from_zip(z, "agent_ids.pkl")
                self.observations = self._load_numpy_buffer_from_zip(
                    z, "observations.npy"
                )
                self.actions = self._load_numpy_buffer_from_zip(
                    z, "actions.npy", convert_dtype=False
                )
                self.log_probs = self._load_tensor_buffer_from_zip(z, "log_probs.pt")
                self.values = self._load_tensor_buffer_from_zip(z, "values.pt")
                self.advantages = self._load_tensor_buffer_from_zip(z, "advantages.pt")
        except FileNotFoundError:
            print(
                f"{self.config.agent_controller_name}: Tried to load experience buffer from checkpoint using the file at location {os.path.join(self.config.checkpoint_load_folder, EXPERIENCE_BUFFER_FILE)}, but there is no such file! A blank experience buffer will be used instead."
            )

    def _load_numpy_buffer_from_zip(
        self, z: zipfile.ZipFile, filename: str, convert_dtype: bool = True
    ) -> NumpyCircularBuffer:
        numpy_buffer = NumpyCircularBuffer(
            capacity=self.config.experience_buffer_config.max_size,
            forced_dtype=torch_to_numpy_dtype(self.config.dtype)
            if convert_dtype
            else None,
        )
        if filename in z.namelist():
            loaded_data = np.load(BytesIO(z.read(filename)), allow_pickle=False)
            numpy_buffer.append(loaded_data)
        return numpy_buffer

    @staticmethod
    def _save_numpy_buffer_to_zip(
        z: zipfile.ZipFile, filename: str, v: NumpyCircularBuffer
    ):
        arr = v.array()
        if arr is not None:
            buf = BytesIO()
            np.save(
                buf,
                arr,
                allow_pickle=False,
            )
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
                NumpyExperienceBuffer._save_numpy_buffer_to_zip(
                    z, "observations.npy", self.observations
                )
                NumpyExperienceBuffer._save_numpy_buffer_to_zip(
                    z, "actions.npy", self.actions
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
        exp_buffer_data, trajectory_processor_data = (
            self.trajectory_processor.process_trajectories(trajectories)
        )
        (agent_ids, observations, actions, log_probs, values, advantages) = (
            exp_buffer_data
        )

        self.agent_ids = _cat_list(
            self.agent_ids, agent_ids, self.config.experience_buffer_config.max_size
        )
        self.observations.append(
            np.array(observations),
        )
        self.actions.append(np.array(actions))
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
        py_indices: list[int] = indices.tolist()
        obs_arr = self.observations.array()
        actions_arr = self.actions.array()
        log_probs_tensor = self.log_probs.tensor()
        values_tensor = self.values.tensor()
        advantages_tensor = self.advantages.tensor()
        assert (
            obs_arr is not None
            and actions_arr is not None
            and log_probs_tensor is not None
            and values_tensor is not None
            and advantages_tensor is not None
        ), "Cannot get samples of experience buffer before any data has been submitted"
        return (
            [self.agent_ids[index] for index in py_indices],
            obs_arr[indices],
            actions_arr[indices],
            log_probs_tensor[indices],
            values_tensor[indices],
            advantages_tensor[indices],
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
        total_samples = len(self.agent_ids)
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
        self.observations = NumpyCircularBuffer(
            capacity=self.max_size, forced_dtype=torch_to_numpy_dtype(self.config.dtype)
        )
        self.actions = NumpyCircularBuffer(capacity=self.max_size, forced_dtype=None)
        self.log_probs = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
            forced_dtype=self.config.dtype,
        )
        self.values = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
            forced_dtype=self.config.dtype,
        )
        self.advantages = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
            forced_dtype=self.config.dtype,
        )
