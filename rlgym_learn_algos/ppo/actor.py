# pyright: reportUnusedParameter=false

from collections.abc import Iterable, Sequence
from typing import Any, Generic

import torch.nn as nn
from rlgym.api import ActionType, AgentID, ObsType
from torch import Tensor


class Actor(nn.Module, Generic[AgentID, ObsType, ActionType]):
    def __init__(self):
        super().__init__()

    def get_action(
        self,
        agent_id_list: list[AgentID],
        obs_list: list[ObsType],
        **kwargs: dict[str, Any],
    ) -> tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: tuple of a list of chosen actions and Tensor with shape (n,) of log probs (float32), with the action list and the first (only) dimension of the tensor parallel with obs_list.
        """
        raise NotImplementedError

    def get_backprop_data(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[ObsType],
        acts: Sequence[ActionType],
        **kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        """
        Function to compute the data necessary for backpropagation.
        :param agent_id_list: list of agent ids, parallel with obs_list.
        :param obs_list: list of ObsTypes to pass through the policy
        :param acts: Actions taken by the policy, parallel with obs_list
        :return: (Action log probs tensor with first dimension parallel with acts, mean entropy as 0-dimensional tensor).
        """
        raise NotImplementedError
