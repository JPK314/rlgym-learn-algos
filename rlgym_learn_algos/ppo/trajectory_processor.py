from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import BaseModel, InstanceOf
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor, device, dtype

from .trajectory import Trajectory

TrajectoryProcessorConfig = TypeVar(
    "TrajectoryProcessorConfig", bound=InstanceOf[BaseModel]
)
TrajectoryProcessorData = TypeVar("TrajectoryProcessorData")

TRAJECTORY_PROCESSOR_FILE = "trajectory_processor.json"


@dataclass
class DerivedTrajectoryProcessorConfig(Generic[TrajectoryProcessorConfig]):
    trajectory_processor_config: Optional[TrajectoryProcessorConfig]
    agent_controller_name: str
    dtype: dtype
    device: device
    checkpoint_load_folder: Optional[str] = None


class TrajectoryProcessor(
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        TrajectoryProcessorData,
    ]
):
    @property
    def config_model(self) -> type[Optional[TrajectoryProcessorConfig]]:
        """
        Function to return the config model type that your TrajectoryProcessor implementation uses. Defaults to NoneType.
        """
        return type(None)

    @abstractmethod
    def process_trajectories(
        self,
        trajectories: List[Trajectory[AgentID, ActionType, ObsType, RewardType]],
    ) -> Tuple[
        Tuple[List[AgentID], List[ObsType], List[ActionType], Tensor, Tensor, Tensor],
        TrajectoryProcessorData,
    ]:
        """
        :param trajectories: List of Trajectory instances from which to generate experience.
        :return: Tuple of (Tuple of parallel lists (considering tensors as a list in their first dimension)
            with (AgentID, ObsType), ActionType, log prob, value, and advantage respectively) and
            TrajectoryProcessorData (for use in the MetricsLogger).
            log prob, value, and advantage tensors should be with dtype=dtype and device=device.
        """
        raise NotImplementedError

    def load(self, config: DerivedTrajectoryProcessorConfig[TrajectoryProcessorConfig]):
        pass

    def save_checkpoint(self, folder_path):
        pass
