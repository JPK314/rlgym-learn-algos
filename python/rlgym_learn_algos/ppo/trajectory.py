from dataclasses import dataclass
from typing import Generic

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Trajectory(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (  # pyright: ignore [reportUnannotatedClassAttribute]
        "action_list",
        "agent_id",
        "final_obs",
        "final_val_pred",
        "log_probs",
        "obs_list",
        "reward_list",
        "truncated",
        "val_preds",
    )
    agent_id: AgentID
    obs_list: list[ObsType]
    action_list: list[ActionType]
    log_probs: Tensor
    reward_list: list[RewardType]
    val_preds: Tensor | None  # must be on cpu
    final_obs: ObsType | None
    final_val_pred: Tensor | None  # must be on cpu
    truncated: bool
