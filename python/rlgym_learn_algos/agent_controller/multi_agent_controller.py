# pyright: reportUnusedParameter=false

from typing import Generic

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn import EnvActionResponse
from rlgym_learn.api import (
    AgentController,
    AgentControllerConfig,
)


class MultiAgentController(
    AgentController[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    Generic[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    def choose_agents(self, agent_id_list: list[AgentID]) -> list[int]:
        """
        Function to determine which agent ids (and their associated observations) this agent controller
        will return the actions (and their associated log probs) for.
        :param agent_id_list: List of AgentIDs available to decide actions for
        :return: list of indices from the agent_id_list which will be used to call get_actions for this agent_controller. If the last agent controller fails to select all agent ids,
        meaning none of the agent controllers chose at least one agent id, an exception is thrown.
        """
        return []

    def process_env_actions(
        self, env_actions: dict[str, EnvActionResponse[AgentID, StateType]]
    ):
        """
        Function to process the env actions that will be used by environments.
        :param env_actions: Dictionary with environment ids as keys and EnvActionResponse as values. These will not be None, and all environment ids which the agent manager is currently getting actions for will be present in the dictionary. Note that if there are multiple agent controllers, there may be more entries than were present in the state_info dict received in choose_env_actions.

        It may cause undefined behavior to modify this dict.
        """
        pass
