from rlgym.api import ActionType, AgentID, ObsType

from .actor import Actor
from .critic import Critic


def log_actor_critic_parameter_counts(
    actor: Actor[AgentID, ObsType, ActionType],
    critic: Critic[AgentID, ObsType],
    agent_controller_name: str | None = None,
):
    # Calculate parameter counts
    actor_params = actor.parameters()
    critic_params = critic.parameters()

    trainable_actor_parameters = filter(lambda p: p.requires_grad, actor_params)
    actor_params_count = sum(p.numel() for p in trainable_actor_parameters)

    trainable_critic_parameters = filter(lambda p: p.requires_grad, critic_params)
    critic_params_count = sum(p.numel() for p in trainable_critic_parameters)

    total_parameters = actor_params_count + critic_params_count

    # Display in a structured manner
    log_prefix = (
        f"{agent_controller_name}:" if agent_controller_name is not None else ""
    )
    print(f"{log_prefix} Trainable Parameters:")
    print(f"{log_prefix} {'Component':<10} {'Count':<10}")
    print("-" * 20)
    print(f"{log_prefix} {'Policy':<10} {actor_params_count:<10}")
    print(f"{log_prefix} {'Critic':<10} {critic_params_count:<10}")
    print("-" * 20)
    print(f"{log_prefix} {'Total':<10} {total_parameters:<10}")
