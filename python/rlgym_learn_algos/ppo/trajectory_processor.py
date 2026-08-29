from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Generic

from pydantic import BaseModel, InstanceOf
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor, device, dtype
from typing_extensions import TypeVar

from .trajectory import Trajectory

TrajectoryProcessorConfig = TypeVar(
    "TrajectoryProcessorConfig", bound=InstanceOf[BaseModel] | None
)
TrajectoryProcessorData = TypeVar("TrajectoryProcessorData")

TRAJECTORY_PROCESSOR_FILE = "trajectory_processor.json"


@dataclass
class DerivedTrajectoryProcessorConfig(Generic[TrajectoryProcessorConfig]):
    trajectory_processor_config: TrajectoryProcessorConfig
    agent_controller_name: str
    dtype: dtype
    device: device
    seed: int
    checkpoint_load_folder: str | None = None


class TrajectoryProcessor(
    ABC,
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        TrajectoryProcessorData,
    ],
):
    @property
    @abstractmethod
    def config_model(self) -> type[TrajectoryProcessorConfig] | None:
        """
        Function to return the config model type that your TrajectoryProcessor implementation uses, or None if no config model is used.
        """

    @abstractmethod
    def process_trajectories(
        self,
        trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]],
    ) -> tuple[
        tuple[list[AgentID], list[ObsType], list[ActionType], Tensor, Tensor, Tensor],
        TrajectoryProcessorData,
    ]:
        """
        :param trajectories: List of Trajectory instances from which to generate experience.
        :return: Tuple of (Tuple of parallel lists (considering tensors as a list in their first dimension)
            with (AgentID, ObsType), ActionType, log prob, value, and advantage respectively) and
            TrajectoryProcessorData (for use in the MetricsLogger).
            log prob, value, and advantage tensors should be with dtype=dtype and device=device.
        """

    @abstractmethod
    def load(
        self, config: DerivedTrajectoryProcessorConfig[TrajectoryProcessorConfig]
    ): ...

    @abstractmethod
    def save_checkpoint(self, folder_path: str | PathLike[str]): ...
