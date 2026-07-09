from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, InstanceOf
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn import BaseConfigModel, EnvAction, ProcessConfigModel, Timestep
from rlgym_learn.api import AgentControllerConfig, DerivedAgentControllerConfig

MultiAgentSubcontrollerConfig = TypeVar(
    "MultiAgentSubcontrollerConfig", bound=InstanceOf[BaseModel] | None
)


@dataclass
class DerivedMultiAgentSubcontrollerConfig(
    Generic[
        MultiAgentSubcontrollerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    subcontroller_mode: bool
    subcontroller_name: str
    subcontroller_config: MultiAgentSubcontrollerConfig
    base_config: BaseConfigModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
    process_config: ProcessConfigModel
    save_folder: str

    @staticmethod
    def from_agent_controller_config(
        config: DerivedAgentControllerConfig[
            AgentControllerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
        subcontroller_name: str,
    ) -> DerivedMultiAgentSubcontrollerConfig[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]:
        return DerivedMultiAgentSubcontrollerConfig(
            subcontroller_mode=False,
            subcontroller_name=subcontroller_name,
            subcontroller_config=config.agent_controller_config,
            base_config=config.base_config,
            process_config=config.process_config,
            save_folder=config.save_folder,
        )

    def to_agent_controller_config(
        self,
    ) -> DerivedAgentControllerConfig[
        MultiAgentSubcontrollerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]:
        return DerivedAgentControllerConfig(
            agent_controller_config=self.subcontroller_config,
            base_config=self.base_config,
            process_config=self.process_config,
            save_folder=self.save_folder,
        )


class MultiAgentSubcontroller(
    ABC,
    Generic[
        MultiAgentSubcontrollerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    @property
    @abstractmethod
    def config_model(
        self,
    ) -> type[MultiAgentSubcontrollerConfig] | None:
        """
        Function to return the config model type that your MultiAgentSubcontroller implementation uses, or None if no config model is used.
        """

    @abstractmethod
    def get_actions(
        self,
        env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]],
    ) -> Mapping[int, Iterable[ActionType]]:
        """
        Function to get actions for agents based on agent ids and observations.
        :param env_obs_data_dict: Dict with env_ids as keys and, for each env_id, a tuple of parallel lists of AgentIDs and ObsTypes for each agent that needs an action from this agent controller.
        :return: For each env_id in env_obs_data_dict, an iterable parallel with the AgentID and ObsType lists containing the ActionType chosen for each agent.
        """

    @abstractmethod
    def process_env_actions(
        self, env_actions: dict[int, EnvAction[AgentID, ActionType, StateType]]
    ):
        """
        Function to process the env actions that will be used by environments.
        :param env_actions: Dictionary with environment ids as keys and EnvAction as values. Note that if there are multiple agent controllers, there may be more entries than were present in the env_obs_data_dict dict received in get_actions.

        Modifying env_actions is undefined behavior.
        """

    @abstractmethod
    def process_timestep_data(
        self,
        timestep_data: dict[
            int,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
    ):
        """
        Function to handle processing of timesteps.
        :param timestep_data: Dictionary with environment ids as keys and tuples of:

        timesteps from the environment (the order of agent ids in this list is fixed until a reset or set_state EnvAction is performed),

        shared info for the environment (if shared_info_serde_type is non-None),

        and the state (if the previous EnvAction for this environment id had send_state=True).
        """

    @abstractmethod
    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        """
        Function to handle managing any state related to space types. Called once before load, may be called at other points according to the env action types.
        """

    @abstractmethod
    def subcontroller_load(
        self,
        config: DerivedMultiAgentSubcontrollerConfig[
            MultiAgentSubcontrollerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ):
        """
        Function to load the subcontroller. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent subcontroller specific config.

        If you are implementing a class that is a subclass of both MultiAgentSubcontroller and AgentController, it is recommended to construct an instance of
        DerivedMultiAgentSubcontrollerConfig from the DerivedAgentControllerConfig and pass it to this method instead of impelementing loading twice.
        """

    @abstractmethod
    def save_checkpoint(self):
        """
        Function to save a checkpoint of the agent.
        """

    @abstractmethod
    def cleanup(self):
        """
        Function to clean up any memory still in use when shutting down.
        """
