import os
import pickle
import zipfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Generic, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)

from ..util.circular_buffers import TensorCircularBuffer
from ..util.torch_pydantic import PydanticTorchDevice
from .trajectory import Trajectory
from .trajectory_processor import (
    DerivedTrajectoryProcessorConfig,
    TrajectoryProcessor,
    TrajectoryProcessorConfig,
    TrajectoryProcessorData,
)

EXPERIENCE_BUFFER_FILE = "experience_buffer.zip"


class ExperienceBufferConfigModel(
    BaseModel, Generic[TrajectoryProcessorConfig], extra="forbid"
):
    max_size: int = 100000
    device: PydanticTorchDevice = Field(default="cpu", validate_default=True)
    save_experience_buffer_in_checkpoint: bool = True
    trajectory_processor_config: TrajectoryProcessorConfig = None  # pyright: ignore [reportAssignmentType]

    @model_validator(mode="before")
    @classmethod
    def validate_trajectory_processor_config_model(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        experience_buffer: (
            ExperienceBuffer[
                TrajectoryProcessorConfig,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
            ]
            | None
        ) = info.context
        data_dict = data
        if (
            experience_buffer is not None
            and isinstance(data_dict, dict)
            and "trajectory_processor_config" in data_dict
        ):
            data_dict = cast(dict[Any, Any], data_dict)
            trajectory_processor_config_raw = data_dict["trajectory_processor_config"]
            if isinstance(trajectory_processor_config_raw, dict):
                trajectory_processor_config_model_type = (
                    experience_buffer.trajectory_processor.config_model
                )
                if trajectory_processor_config_model_type is None:
                    trajectory_processor_config = None
                else:
                    trajectory_processor_config = cast(
                        BaseModel, trajectory_processor_config_model_type
                    ).model_validate(
                        trajectory_processor_config_raw,
                        context=experience_buffer.trajectory_processor,
                    )
            else:
                trajectory_processor_config = trajectory_processor_config_raw
            data_dict["trajectory_processor_config"] = trajectory_processor_config
        return data


@dataclass
class DerivedExperienceBufferConfig(
    Generic[TrajectoryProcessorConfig, ObsSpaceType, ActionSpaceType]
):
    experience_buffer_config: ExperienceBufferConfigModel[TrajectoryProcessorConfig]
    agent_controller_name: str
    obs_space: ObsSpaceType
    action_space: ActionSpaceType
    seed: int
    dtype: torch.dtype
    checkpoint_load_folder: str | None = None


class ExperienceBuffer(
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        TrajectoryProcessorData,
    ]
):
    @staticmethod
    def _cat_list(cur: list[Any], new: list[Any], size: int):
        new_len = len(new)
        if new_len > size:
            t = new[-size:]
        elif new_len == size:
            t = new
        elif len(cur) + new_len > size:
            t = cur[new_len - size :] + new
        else:
            t = cur + new
        return t

    def __init__(
        self,
        trajectory_processor: TrajectoryProcessor[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ],
    ):
        self.trajectory_processor: TrajectoryProcessor[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ] = trajectory_processor
        self.agent_ids: list[AgentID] = []
        self.observations: list[ObsType] = []
        self.actions: list[ActionType] = []
        self.log_probs: TensorCircularBuffer
        self.values: TensorCircularBuffer
        self.advantages: TensorCircularBuffer
        self.rng: np.random.RandomState = np.random.RandomState(0)
        self.max_size: int
        self.config: DerivedExperienceBufferConfig[
            TrajectoryProcessorConfig, ObsSpaceType, ActionSpaceType
        ]

    def load(
        self,
        config: DerivedExperienceBufferConfig[
            TrajectoryProcessorConfig, ObsSpaceType, ActionSpaceType
        ],
    ):
        self.config = config
        self.max_size = config.experience_buffer_config.max_size
        self.rng = np.random.RandomState(config.seed)
        self.trajectory_processor.load(
            DerivedTrajectoryProcessorConfig(
                trajectory_processor_config=config.experience_buffer_config.trajectory_processor_config,
                agent_controller_name=config.agent_controller_name,
                seed=config.seed,
                dtype=config.dtype,
                device=config.experience_buffer_config.device,
            )
        )
        self.log_probs = TensorCircularBuffer(
            capacity=self.max_size,
            device=config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.values = TensorCircularBuffer(
            capacity=self.max_size,
            device=config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.advantages = TensorCircularBuffer(
            capacity=self.max_size,
            device=config.experience_buffer_config.device,
            pin_memory=True,
        )
        if self.config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

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
                self.observations = self._load_list_from_zip(z, "observations.pkl")
                self.actions = self._load_list_from_zip(z, "actions.pkl")
                self.log_probs = self._load_tensor_buffer_from_zip(z, "log_probs.pt")
                self.values = self._load_tensor_buffer_from_zip(z, "values.pt")
                self.advantages = self._load_tensor_buffer_from_zip(z, "advantages.pt")
        except FileNotFoundError:
            print(
                f"{self.config.agent_controller_name}: Tried to load experience buffer from checkpoint using the file at location {os.path.join(self.config.checkpoint_load_folder, EXPERIENCE_BUFFER_FILE)}, but there is no such file! A blank experience buffer will be used instead."
            )

    def _load_list_from_zip(self, z: zipfile.ZipFile, filename: str) -> list[Any]:
        loaded_data = pickle.loads(z.read(filename))
        loaded_len = len(loaded_data)
        if loaded_len > self.config.experience_buffer_config.max_size:
            print(
                f"{self.config.agent_controller_name}: Experience buffer checkpoint length for {filename} was {loaded_len}, but the configured capacity is {self.config.experience_buffer_config.max_size}. The newest samples that fit will be retained."
            )
            ret_list = loaded_data[-self.config.experience_buffer_config.max_size :]
        else:
            ret_list = loaded_data
        return ret_list

    def _load_tensor_buffer_from_zip(
        self,
        z: zipfile.ZipFile,
        filename: str,
    ) -> TensorCircularBuffer:
        tensor_buffer = TensorCircularBuffer(
            capacity=self.config.experience_buffer_config.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
        if filename in z.namelist():
            state = torch.load(
                BytesIO(z.read(filename)),
                map_location=self.config.experience_buffer_config.device,
                weights_only=True,
            )
            loaded_data: torch.Tensor = state["data"].to(self.config.dtype)
            loaded_capacity: int = state["capacity"]
            if loaded_capacity != self.config.experience_buffer_config.max_size:
                print(
                    f"{self.config.agent_controller_name}: Experience buffer checkpoint capacity for {filename} was {loaded_capacity}, but the configured capacity is {self.config.experience_buffer_config.max_size}. The newest samples that fit will be retained."
                )
            tensor_buffer.append(loaded_data)
        return tensor_buffer

    @staticmethod
    def _save_tensor_buffer_to_zip(
        z: zipfile.ZipFile, filename: str, v: TensorCircularBuffer
    ):
        t = v.tensor()
        if t is not None:
            buf = BytesIO()
            torch.save({"capacity": v.capacity, "data": t.detach().cpu()}, buf)
            z.writestr(filename, buf.getvalue())

    def save_checkpoint(self, folder_path: str | os.PathLike[str]):
        os.makedirs(folder_path, exist_ok=True)
        if self.config.experience_buffer_config.save_experience_buffer_in_checkpoint:
            with zipfile.ZipFile(
                os.path.join(folder_path, EXPERIENCE_BUFFER_FILE),
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as z:
                z.writestr("agent_ids.pkl", pickle.dumps(self.agent_ids))
                z.writestr("observations.pkl", pickle.dumps(self.observations))
                z.writestr("actions.pkl", pickle.dumps(self.actions))
                ExperienceBuffer._save_tensor_buffer_to_zip(
                    z, "log_probs.pt", self.log_probs
                )
                ExperienceBuffer._save_tensor_buffer_to_zip(z, "values.pt", self.values)
                ExperienceBuffer._save_tensor_buffer_to_zip(
                    z, "advantages.pt", self.advantages
                )
        self.trajectory_processor.save_checkpoint(folder_path)

    def submit_experience(
        self, trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]]
    ) -> TrajectoryProcessorData:
        """
        Function to add experience to the buffer.

        :param trajectories: A list of Trajectory instances to process and store as experience in the buffer.

        :return: TrajectoryProcessorData - statistics intended for consumption by the caller (likely in a metrics logger) from the trajectory processor.
        """

        _cat_list = ExperienceBuffer._cat_list
        exp_buffer_data, trajectory_processor_data = (
            self.trajectory_processor.process_trajectories(trajectories)
        )
        (agent_ids, observations, actions, log_probs, values, advantages) = (
            exp_buffer_data
        )

        self.agent_ids = _cat_list(self.agent_ids, agent_ids, self.max_size)
        self.observations = _cat_list(
            self.observations,
            observations,
            self.max_size,
        )
        self.actions = _cat_list(self.actions, actions, self.max_size)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.advantages.append(advantages)

        return trajectory_processor_data

    def _get_samples(
        self, indices: NDArray[np.int64]
    ) -> tuple[
        Sequence[AgentID],
        Sequence[ObsType],
        Sequence[ActionType],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        py_indices: list[int] = indices.tolist()
        log_probs_tensor = self.log_probs.tensor()
        values_tensor = self.values.tensor()
        advantages_tensor = self.advantages.tensor()
        assert (
            log_probs_tensor is not None
            and values_tensor is not None
            and advantages_tensor is not None
        ), "Cannot get samples of experience buffer before any data has been submitted"
        return (
            [self.agent_ids[index] for index in py_indices],
            [self.observations[index] for index in py_indices],
            [self.actions[index] for index in py_indices],
            log_probs_tensor[indices],
            values_tensor[indices],
            advantages_tensor[indices],
        )

    def get_all_batches_shuffled(
        self, batch_size: int
    ) -> Generator[
        tuple[
            Sequence[AgentID],
            Sequence[ObsType],
            Sequence[ActionType],
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
        self.observations = []
        self.actions = []
        self.log_probs = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.values = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
        self.advantages = TensorCircularBuffer(
            capacity=self.max_size,
            device=self.config.experience_buffer_config.device,
            pin_memory=True,
        )
