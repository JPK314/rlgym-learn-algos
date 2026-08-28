from typing import Generic, TypedDict

from numpy import ndarray
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor

from ...ppo import Trajectory
from ...ppo.trajectory_processor import (
    DerivedTrajectoryProcessorConfig,
    TrajectoryProcessorConfig,
)
from ...stateful_functions import BatchRewardTypeNumpyConverter

class WelfordRunningStatsStateDict(TypedDict):
    mean: float
    count: int
    m2: float

class GAETrajectoryProcessor(Generic[AgentID, ObsType, ActionType, RewardType]):
    def __new__(
        cls,
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[RewardType],
    ) -> GAETrajectoryProcessor[AgentID, ObsType, ActionType, RewardType]: ...
    def load(
        self,
        config: DerivedTrajectoryProcessorConfig[TrajectoryProcessorConfig],
    ) -> None: ...
    def load_state_dict(self, state_dict: WelfordRunningStatsStateDict) -> None: ...
    def process_trajectories(
        self,
        trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]],
    ) -> tuple[
        list[AgentID],
        list[ObsType],
        list[ActionType],
        Tensor,
        Tensor,
        ndarray,
        ndarray,
        float,
        float,
    ]: ...
    def state_dict(self) -> WelfordRunningStatsStateDict | None: ...
