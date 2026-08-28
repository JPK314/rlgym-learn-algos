from collections.abc import Iterable, Sequence
from typing import Any, Generic

from rlgym.api import ActionType, AgentID, ObsType
from torch import Tensor, nn
from typing_extensions import override

from .actor import Actor
from .actor_critic import ActorCritic
from .critic import Critic


class SeparateActorCritic(
    ActorCritic[AgentID, ObsType, ActionType], Generic[AgentID, ObsType, ActionType]
):
    def __init__(
        self,
        actor: Actor[AgentID, ObsType, ActionType],
        critic: Critic[AgentID, ObsType],
        agent_controller_name: str | None = None,
    ):
        super().__init__()
        self.actor: Actor[AgentID, ObsType, ActionType] = actor
        self.critic: Critic[AgentID, ObsType] = critic

    @override
    def get_actions(
        self,
        agent_id_list: list[AgentID],
        obs_list: list[ObsType],
        **kwargs: dict[str, Any],
    ) -> tuple[Iterable[ActionType], Tensor]:
        return self.actor.get_actions(agent_id_list, obs_list, **kwargs)

    @override
    def get_value_predictions(
        self, agent_id_list: list[AgentID], obs_list: list[ObsType]
    ):
        return self.critic(agent_id_list, obs_list)

    @override
    def get_backprop_data(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[ObsType],
        action_list: Sequence[ActionType],
        **kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        (log_probs, entropy) = self.actor.get_backprop_data(
            agent_id_list, obs_list, action_list, **kwargs
        )
        vals = self.critic(agent_id_list, obs_list)
        return (log_probs, entropy, vals)

    @override
    def get_actor_parameter_vector(self) -> Tensor:
        return nn.utils.parameters_to_vector(self.actor.parameters())

    @override
    def get_critic_parameter_vector(self) -> Tensor:
        return nn.utils.parameters_to_vector(self.critic.parameters())
