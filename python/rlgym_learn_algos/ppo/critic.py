from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from rlgym.api import AgentID, ObsType
from torch import Tensor, nn
from typing_extensions import override


class Critic(ABC, nn.Module, Generic[AgentID, ObsType]):
    def __init__(self):
        super().__init__()

    @override
    @abstractmethod
    def forward(
        self, agent_id_list: Sequence[AgentID], obs_list: Sequence[ObsType]
    ) -> Tensor:
        """
        :param agent_id_list: List of AgentIDs, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes to compute values for.
        :return: Tensor. Must be 0-dimensional for PPO, with dtype float32.
        """
