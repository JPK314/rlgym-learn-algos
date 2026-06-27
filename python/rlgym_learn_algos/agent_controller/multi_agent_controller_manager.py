from collections.abc import Mapping
from typing import Any, Generic, cast

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    model_validator,
)
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn import AnyBaseModel
from rlgym_learn.api import AgentController
from typing_extensions import Self, override

from ._rlgym_learn_algos.agent_controller import (
    MultiAgentControllerCoordinator as RustMultiAgentControllerCoordinator,
)
from .multi_agent_controller import MultiAgentController


class MultiAgentControllerCoordinatorConfig(
    BaseModel,
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    extra="forbid",
):
    agent_controller_config: dict[str, AnyBaseModel | None] = Field(
        default_factory=dict
    )
    agent_controllers_save_folder: str = "agent_controllers_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def validate_agent_controllers_config_models(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        agent_controllers: (
            dict[
                str,
                AgentController[
                    Any,
                    AgentID,
                    ObsType,
                    ActionType,
                    RewardType,
                    StateType,
                    ObsSpaceType,
                    ActionSpaceType,
                ],
            ]
            | None
        ) = info.context
        data_dict = data
        data_config_model = data
        if agent_controllers is not None:
            if isinstance(data_dict, dict) and "agent_controllers_config" in data:
                data_dict = cast(dict[Any, Any], data_dict)
                agent_controllers_config_raw = data_dict["agent_controllers_config"]
                agent_controllers_config: dict[str, BaseModel | None] = {}
                for k, v in agent_controllers_config_raw.items():
                    if k in agent_controllers:
                        if isinstance(v, dict):
                            agent_controller = agent_controllers[k]
                            agent_controller_config_model_type = (
                                agent_controller.config_model
                            )
                            if agent_controller_config_model_type is None:
                                agent_controllers_config[k] = None
                            else:
                                agent_controllers_config[k] = cast(
                                    BaseModel, agent_controller_config_model_type
                                ).model_validate(v, context=agent_controller)

                        else:
                            agent_controllers_config[k] = v
                data_dict["agent_controllers_config"] = agent_controllers_config
            elif isinstance(data_config_model, LearningCoordinatorConfigModel):
                data_config_model.agent_controllers_config = {
                    k: v
                    for k, v in data_config_model.agent_controllers_config.items()
                    if k in agent_controllers
                }
        return data

    @model_validator(mode="after")
    def validate_agent_controllers_all_present(self, info: ValidationInfo) -> Self:
        agent_controllers: (
            dict[
                str,
                AgentController[
                    Any,
                    AgentID,
                    ObsType,
                    ActionType,
                    RewardType,
                    StateType,
                    ObsSpaceType,
                    ActionSpaceType,
                ],
            ]
            | None
        ) = info.context
        if agent_controllers is not None:
            agent_controller_keys_not_in_config = [
                v for v in agent_controllers if v not in self.agent_controllers_config
            ]
            assert len(agent_controller_keys_not_in_config) == 0, (
                f"some agent controllers do not have keys present in agent_controllers_config. The following keys from agent_controllers are not present in agent_controllers_config: {agent_controller_keys_not_in_config}"
            )
        return self


class MultiAgentControllerCoordinator(
    AgentController[
        MultiAgentControllerCoordinatorConfig[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    agent_controllers: Mapping[
        str,
        AgentController[
            Any,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ]

    def __init__(
        self,
        agent_controllers: Mapping[
            str,
            MultiAgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ],
        ],
    ):
        self.agent_controllers = agent_controllers
        self.rust_multi_agent_controller_coordinator: RustMultiAgentControllerCoordinator[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = RustMultiAgentControllerCoordinator(
            self.agent_controllers_list, batched_tensor_action_associated_learning_data
        )

    @property
    @override
    def config_model(
        self,
    ) -> (
        type[
            MultiAgentControllerCoordinatorConfig[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
        ]
        | None
    ):
        return MultiAgentControllerCoordinatorConfig

    def get_actions(
        self,
        agent_id_list: list[AgentID],
        obs_list: list[ObsType],
    ) -> tuple[Iterable[ActionType], Any]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of a list of chosen actions and action associated learning data.

        If base_config.batched_tensor_action_associated_learning_data is true, the action associated learning data should be a tensor with the first dimension parallel with the action list. Otherwise, the action associated learning data should be a list parallel with the action list.
        """
        raise NotImplementedError

    def process_timestep_data(
        self,
        timestep_data: dict[
            str,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                ActionAssociatedLearningData | None,
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
    ):
        """
        Function to handle processing of timesteps.
        :param timestep_data: Dictionary with environment ids as keys and tuples of:

        timesteps from the environment (the order of agent ids in this list is fixed until a reset or set_state env action is taken),

        action associated learning data (parallel to the timestep list, and None if no timesteps exist for the environment),

        shared info for the environment (if shared_info_serde_type is non-None),

        and the state (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True).

        Do not modify this dict as it will be passed by reference to other agent controllers.
        """
        pass

    def choose_env_actions(
        self,
        state_info: dict[
            str,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> dict[str, EnvActionResponse[AgentID, StateType] | None]:
        """
        Function to choose EnvActionResponse per environment based on environment information. Called after process_timestep_data.
        :param state_info: Dictionary with environment ids as keys and tuples of shared info (if shared_info_serde_type is non-None), StateType (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True), the present terminated dict for the env (None if env was just reset), and the present truncated dict for the env (None if env was just reset).
        :return: Dictionary with environment ids as keys and EnvActionResponse as values. If STEP_RESPONSE is sent for an environment (and the agent manager agrees to use step as the env action for that environment),
        then choose_agents and get_actions will be called asking for the actions for the agents in those environments.
        If None is used as a value in the returned dict, or an environment id key from the state_info dict is not present in the returned dict, the agent manager will ask the other agent controllers for the env action for that environment.
        If all agent controllers have been asked and an environment id is without an env action, an exception is thrown.
        """
        return {}

    def process_env_actions(
        self, env_actions: dict[str, EnvActionResponse[AgentID, StateType]]
    ):
        """
        Function to process the env actions that will be used by environments.
        :param env_actions: Dictionary with environment ids as keys and EnvActionResponse as values. These will not be None, and all environment ids which the agent manager is currently getting actions for will be present in the dictionary. Note that if there are multiple agent controllers, there may be more entries than were present in the state_info dict received in choose_env_actions.

        It may cause undefined behavior to modify this dict.
        """
        pass

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        pass

    def load(
        self,
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
    ):
        """
        Function to load the agent. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent controller specific config.
        """
        pass

    def save_checkpoint(self):
        """
        Function to save a checkpoint of the agent.
        """
        pass

    def cleanup(self):
        """
        Function to clean up any memory still in use when shutting down.
        """
        pass
