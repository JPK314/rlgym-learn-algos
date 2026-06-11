# pyright: reportUnusedParameter=false

from typing import Generic

from rlgym.api import AgentID, ObsType


class ObsStandardizer(Generic[AgentID, ObsType]):
    def standardize(
        self, agent_id_list: list[AgentID], obs_list: list[ObsType]
    ) -> list[ObsType]:
        """
        :param agent_id_list: List of AgentIDs, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes to standardize.
        :return: List of standardized observations, parallel with input lists.
        """
        raise NotImplementedError
