from dataclasses import dataclass
from typing import Generic

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Trajectory(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (  # pyright: ignore [reportUnannotatedClassAttribute]
        "agent_id",
        "obs_list",
        "action_list",
        "log_probs",
        "reward_list",
        "val_preds",
        "final_obs",
        "final_val_pred",
        "truncated",
    )
    agent_id: AgentID
    obs_list: list[ObsType]
    action_list: list[ActionType]
    log_probs: Tensor
    reward_list: list[RewardType]
    val_preds: Tensor | None
    final_obs: ObsType | None
    final_val_pred: Tensor | None
    truncated: bool
