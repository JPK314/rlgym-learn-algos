from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Generic

from rlgym.api import ActionType, AgentID, ObsType
from torch import Tensor, nn


class ActorCritic(ABC, nn.Module, Generic[AgentID, ObsType, ActionType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_actions(
        self,
        agent_id_list: list[AgentID],
        obs_list: list[ObsType],
        **kwargs: dict[str, Any],
    ) -> tuple[Iterable[ActionType], Tensor]:
        """
        Function to get actions and the log of their probabilities from the policy given observations.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of (Iterable of chosen actions, Tensor with shape (n,) of log probs (float32) with the action list and the first (only) dimension of the tensor parallel with obs_list).
        """

    @abstractmethod
    def get_value_predictions(
        self, agent_id_list: list[AgentID], obs_list: list[ObsType]
    ) -> Tensor:
        """
        Function to get value predictions from the critic given observations.
        :param agent_id_list: List of AgentIDs, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes to compute values for.
        :return: Tensor. Must be 1-dimensional (parallel to obs_list) for PPO, with dtype float32.
        """

    @abstractmethod
    def get_backprop_data(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[ObsType],
        action_list: Sequence[ActionType],
        **kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Function to compute the data necessary for backpropagation.
        :param agent_id_list: list of agent ids, parallel with obs_list.
        :param obs_list: list of ObsTypes to pass through the policy
        :param action_list: Actions taken by the policy, parallel with obs_list
        :return: Tuple of (Action log probs tensor with first dimension parallel with action_list, mean entropy as 0-dimensional tensor, critic value predictions with first dimension parallel to obs_list).
        """

    @abstractmethod
    def get_actor_parameter_vector(self) -> Tensor:
        """
        Function to return the actor's parameters as a vector, for computing update magnitude.
        :return: Actor model's parameters as a vector.
        """

    @abstractmethod
    def get_critic_parameter_vector(self) -> Tensor:
        """
        Function to return the critic's parameters as a vector, for computing update magnitude.
        :return: Critic model's parameters as a vector.
        """
