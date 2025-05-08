from __future__ import annotations

from typing import TYPE_CHECKING, Generic, List, Tuple

from numpy import dtype, ndarray
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
        cls, batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter
    ) -> GAETrajectoryProcessor: ...
    def load(self, config: DerivedGAETrajectoryProcessorConfig): ...
    def process_trajectories(
        self,
        trajectories: List[Trajectory[AgentID, ObsType, ActionType, RewardType]],
        return_std: ndarray,
    ) -> Tuple[
        List[AgentID],
        List[ObsType],
        List[ActionType],
        Tensor,
        Tensor,
        ndarray,
        ndarray,
        float,
        float,
    ]: ...
