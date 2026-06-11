from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from numpy import dtype, float32, float64, ndarray
from rlgym.api import ActionType, AgentID, ObsType, RewardType

from rlgym_learn_algos.stateful_functions import BatchRewardTypeNumpyConverter

if TYPE_CHECKING:
    from torch import Tensor

    from rlgym_learn_algos.ppo import Trajectory

class DerivedGAETrajectoryProcessorConfig:
    def __new__(
        cls, gamma: float, lmbda: float, dtype: dtype
    ) -> DerivedGAETrajectoryProcessorConfig: ...

class GAETrajectoryProcessor(Generic[AgentID, ObsType, ActionType, RewardType]):
    def __new__(
        cls,
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[RewardType],
    ) -> GAETrajectoryProcessor[AgentID, ObsType, ActionType, RewardType]: ...
    def load(self, config: DerivedGAETrajectoryProcessorConfig) -> None: ...
    def process_trajectories(
        self,
        trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]],
        return_std: float32 | float64,
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
