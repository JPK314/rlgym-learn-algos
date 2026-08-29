import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, cast
from weakref import proxy

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
from rlgym_learn import AnyBaseModel, EnvAction, EnvCloseReason, Timestep
from rlgym_learn.api import AgentController, DerivedAgentControllerConfig
from typing_extensions import Self, override

from ..._rlgym_learn_algos.agent_controller import EnvActionResponse
from ..._rlgym_learn_algos.agent_controller import (
    MultiAgentController as RustMultiAgentController,
)
from .multi_agent_subcontroller import (
    DerivedMultiAgentSubcontrollerConfig,
    MultiAgentSubcontroller,
)


class MultiAgentControllerConfigModel(
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
    subcontrollers_config: dict[str, AnyBaseModel | None] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_controllers_config_models(cls, data: Any, info: ValidationInfo) -> Any:
        multi_agent_controller: (
            MultiAgentController[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
            | None
        ) = info.context
        data_dict = data
        data_config_model = data
        if multi_agent_controller is not None:
            if isinstance(data_dict, dict) and "subcontrollers_config" in data:
                data_dict = cast(dict[Any, Any], data_dict)
                subcontrollers_config_raw = data_dict["subcontrollers_config"]
                subcontrollers_config: dict[str, BaseModel | None] = {}
                for k, v in subcontrollers_config_raw.items():
                    if k in multi_agent_controller.subcontrollers:
                        if isinstance(v, dict):
                            subcontroller = multi_agent_controller.subcontrollers[k]
                            subcontroller_config_model_type = subcontroller.config_model
                            if subcontroller_config_model_type is None:
                                subcontrollers_config[k] = None
                            else:
                                subcontrollers_config[k] = cast(
                                    BaseModel, subcontroller_config_model_type
                                ).model_validate(v, context=subcontroller)

                        else:
                            subcontrollers_config[k] = v
                data_dict["subcontrollers_config"] = subcontrollers_config
            elif isinstance(data_config_model, MultiAgentControllerConfigModel):
                data_config_model.subcontrollers_config = {
                    k: v
                    for k, v in data_config_model.subcontrollers_config.items()
                    if k in multi_agent_controller.subcontrollers
                }
        return data

    @model_validator(mode="after")
    def validate_agent_controllers_all_present(self, info: ValidationInfo) -> Self:
        multi_agent_controller: (
            MultiAgentController[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
            | None
        ) = info.context
        if multi_agent_controller is not None:
            subcontroller_keys_not_in_config = [
                v
                for v in multi_agent_controller.subcontrollers
                if v not in self.subcontrollers_config
            ]
            assert len(subcontroller_keys_not_in_config) == 0, (
                f"some agent subcontrollers do not have keys present in agent_subcontrollers_config. The following keys from agent_subcontrollers are not present in agent_subcontrollers_config: {subcontroller_keys_not_in_config}"
            )
        return self


class MultiAgentController(
    AgentController[
        MultiAgentControllerConfigModel[
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
    ABC,
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
    def __init__(
        self,
        subcontrollers: Mapping[
            str,
            MultiAgentSubcontroller[
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
        self.subcontrollers: Mapping[
            str,
            MultiAgentSubcontroller[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ],
        ] = subcontrollers
        self.subcontrollers_list: list[
            MultiAgentSubcontroller[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
        ] = list(subcontrollers.values())
        self.rust_multi_agent_controller: RustMultiAgentController[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = RustMultiAgentController(proxy(self), subcontrollers)

    @property
    @override
    def config_model(
        self,
    ) -> (
        type[
            MultiAgentControllerConfigModel[
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
        return MultiAgentControllerConfigModel

    @abstractmethod
    def choose_env_actions(
        self,
        env_state_info_dict: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> tuple[int, dict[int, EnvActionResponse[AgentID, StateType]]]:
        """
        Function to choose EnvActionResponse per environment based on environment information.
        :param env_state_info_dict: Dictionary with environment ids as keys and tuples of shared info (if shared_info_serde_type is non-None), StateType (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True), the present terminated dict for the env (None if env was just reset), and the present truncated dict for the env (None if env was just reset).
        :return: Tuple where the first value is the number of new environments to create, and the second value is a dictionary with environment ids as keys and EnvActionResponse instances as values. If a EnvActionResponse.STEP instance is returned for an environment, then delegate_actions will be called for the agents in those environments.
        If any environment id in the state_info dict is not a key in the returned dict, an exception is thrown.

        The default implementation (called using super().choose_env_actions(...)) may be used for convenience, which returns EnvActionResponse.RESET() for an environment id if all agents in that environment are truncated or terminated in the corresponding state info, and returns EnvActionResponse.STEP() otherwise.
        """
        env_action_responses: dict[int, EnvActionResponse[AgentID, StateType]] = {}
        for env_id, (
            _,
            _,
            terminated_dict,
            truncated_dict,
        ) in env_state_info_dict.items():
            if terminated_dict is None or truncated_dict is None:
                # This must be the first env action after a reset, so we step
                env_action_responses[env_id] = EnvActionResponse.STEP()
                continue
            if all(
                terminated or truncated_dict[agent_id]
                for agent_id, terminated in terminated_dict.items()
            ):
                env_action_responses[env_id] = EnvActionResponse.RESET()
                continue
            env_action_responses[env_id] = EnvActionResponse.STEP()
        return (0, env_action_responses)

    @abstractmethod
    def choose_subcontrollers(
        self, agent_ids: dict[int, list[AgentID]]
    ) -> dict[int, list[str]] | None:
        """
        Function to determine which subcontrollers will be responsible for returning actions for given agent ids (and their associated observations).
        :param agent_ids: Dict with env_ids as keys and list of the agent ids available to choose from as values.
        :return: For each env_id, a list of subcontroller names (keys in the agent_subcontrollers dict) parallel to the corresponding AgentID list in agent_ids.
        """

    @override
    def get_env_actions(
        self,
        env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]],
        env_state_info_dict: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> tuple[int, dict[int, EnvAction[AgentID, ActionType, StateType]]]:
        n_new_envs, env_actions = self.rust_multi_agent_controller.get_env_actions(
            env_obs_data_dict, env_state_info_dict
        )
        for subcontroller in self.subcontrollers_list:
            subcontroller.process_env_actions(env_actions)
        return n_new_envs, env_actions

    @override
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
        for subcontroller in self.subcontrollers_list:
            subcontroller.process_timestep_data(timestep_data)

    @override
    def set_space_types(
        self,
        env_spaces_data_dict: dict[
            int, dict[AgentID, tuple[ObsSpaceType, ActionSpaceType]]
        ],
    ):
        for subcontroller in self.subcontrollers_list:
            subcontroller.set_space_types(env_spaces_data_dict)

    @override
    def handle_env_closes(self, env_close_reason_dict: dict[int, EnvCloseReason]):
        for subcontroller in self.subcontrollers_list:
            subcontroller.handle_env_closes(env_close_reason_dict)

    @override
    def load(
        self,
        config: DerivedAgentControllerConfig[
            MultiAgentControllerConfigModel[
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
    ):
        """
        Function to load the agent. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent controller specific config.
        """
        for (
            subcontroller_name,
            subcontroller,
        ) in self.subcontrollers.items():
            subcontroller.subcontroller_load(
                DerivedMultiAgentSubcontrollerConfig(
                    subcontroller_mode=True,
                    subcontroller_name=subcontroller_name,
                    subcontroller_config=config.agent_controller_config.subcontrollers_config[
                        subcontroller_name
                    ],
                    base_config=config.base_config,
                    process_config=config.process_config,
                    save_folder=os.path.join(
                        config.save_folder,
                        subcontroller_name,
                    ),
                )
            )

    @override
    def save_checkpoint(self):
        for subcontroller in self.subcontrollers_list:
            subcontroller.save_checkpoint()

    @override
    def cleanup(self):
        for subcontroller in self.subcontrollers_list:
            subcontroller.cleanup()
