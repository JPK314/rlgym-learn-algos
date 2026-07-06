from collections.abc import Iterable, Mapping
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
from rlgym_learn.api import AgentController, DerivedAgentControllerConfig
from typing_extensions import Self, override

from ._rlgym_learn_algos.agent_controller import (
    MultiAgentController as RustMultiAgentController,
)
from .multi_agent_subcontroller import MultiAgentSubcontroller


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
    agent_subcontrollers_config: dict[str, AnyBaseModel | None] = Field(
        default_factory=dict
    )
    agent_controllers_save_folder: str = "agent_controllers_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def validate_agent_controllers_config_models(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
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
            if isinstance(data_dict, dict) and "agent_controllers_config" in data:
                data_dict = cast(dict[Any, Any], data_dict)
                agent_controllers_config_raw = data_dict["agent_controllers_config"]
                agent_controllers_config: dict[str, BaseModel | None] = {}
                for k, v in agent_controllers_config_raw.items():
                    if k in multi_agent_controller.agent_subcontrollers:
                        if isinstance(v, dict):
                            agent_subcontroller = (
                                multi_agent_controller.agent_subcontrollers[k]
                            )
                            agent_controller_config_model_type = (
                                agent_subcontroller.config_model
                            )
                            if agent_controller_config_model_type is None:
                                agent_controllers_config[k] = None
                            else:
                                agent_controllers_config[k] = cast(
                                    BaseModel, agent_controller_config_model_type
                                ).model_validate(v, context=agent_subcontroller)

                        else:
                            agent_controllers_config[k] = v
                data_dict["agent_controllers_config"] = agent_controllers_config
            elif isinstance(data_config_model, MultiAgentControllerConfigModel):
                data_config_model.agent_subcontrollers_config = {
                    k: v
                    for k, v in data_config_model.agent_subcontrollers_config.items()
                    if k in multi_agent_controller.agent_subcontrollers
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
            agent_subcontroller_keys_not_in_config = [
                v
                for v in multi_agent_controller.agent_subcontrollers
                if v not in self.agent_subcontrollers_config
            ]
            assert len(agent_subcontroller_keys_not_in_config) == 0, (
                f"some agent subcontrollers do not have keys present in agent_subcontrollers_config. The following keys from agent_subcontrollers are not present in agent_subcontrollers_config: {agent_subcontroller_keys_not_in_config}"
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
    agent_subcontrollers: Mapping[
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
    ]

    def __init__(
        self,
        agent_controllers: Mapping[
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
        self.agent_subcontrollers = agent_controllers
        self.rust_multi_agent_controller_coordinator: RustMultiAgentController[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = RustMultiAgentController(
            self.agent_controllers_list, batched_tensor_action_associated_learning_data
        )

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
        pass

    @override
    def save_checkpoint(self):
        """
        Function to save a checkpoint of the agent.
        """
        pass

    @override
    def cleanup(self):
        """
        Function to clean up any memory still in use when shutting down.
        """
        pass
