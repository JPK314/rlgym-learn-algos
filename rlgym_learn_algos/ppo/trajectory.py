from dataclasses import dataclass
from typing import Generic, List

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from tensordict import TensorDict
from torch import Tensor


@dataclass
class Trajectory(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (
        "aald",
        "reward_list",
        "val_preds",
        "final_obs",
        "final_val_pred",
        "truncated",
    )
    aald: TensorDict
    reward_list: List[RewardType]
    val_preds: Tensor
    final_obs: ObsType
    final_val_pred: Tensor
    truncated: bool
