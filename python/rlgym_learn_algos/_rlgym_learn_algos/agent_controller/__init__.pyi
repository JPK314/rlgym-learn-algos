# pyright: reportUnusedParameter=false

# pyright: reportUnusedParameter=false
from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn import EnvAction

if TYPE_CHECKING:
    from ...agent_controller import (
        MultiAgentSubcontroller,
    )
    from ..agent_controller import EnvActionResponse

__all__ = [
    "EnvActionResponse",
]

AgentIDInner = TypeVar("AgentIDInner")
StateTypeInner = TypeVar("StateTypeInner")

@final
class MultiAgentController(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    def __new__(
        cls,
        multi_agent_controller: MultiAgentController[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
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
        ],
    ) -> MultiAgentController[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]: ...
    def get_env_actions(
        self,
        env_obs_data_dict: Mapping[
            int,
            tuple[
                Sequence[AgentID],
                Sequence[ObsType],
            ],
        ],
        env_state_info_dict: Mapping[
            int,
            tuple[
                Mapping[str, Any] | None,
                StateType | None,
                Mapping[AgentID, bool] | None,
                Mapping[AgentID, bool] | None,
            ],
        ],
    ) -> dict[int, EnvAction[AgentID, ActionType, StateType]]: ...

@final
class EnvActionResponseType(Enum):
    STEP = ...
    RESET = ...
    SET_STATE = ...

class EnvActionResponse(Generic[AgentID, StateType]):
    @property
    def enum_type(self) -> EnvActionResponseType: ...
    @property
    def shared_info_setter(self) -> Any | None: ...
    @property
    def desired_state(self) -> Any | None: ...
    @property
    def prev_timestep_id_dict(self) -> Any | None: ...

    @final
    class STEP(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "shared_info_setter_option",
            "send_state",
        )

        def __new__(
            cls,
            shared_info_setter_option: Mapping[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvActionResponse.STEP[AgentIDInner, StateTypeInner]: ...

    @final
    class RESET(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "shared_info_setter_option",
            "send_state",
        )

        def __new__(
            cls,
            shared_info_setter_option: Mapping[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvActionResponse.RESET[AgentIDInner, StateTypeInner]: ...

    @final
    class SET_STATE(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "desired_state",
            "shared_info_setter_option",
            "send_state",
            "prev_timestep_id_dict_option",
        )

        def __new__(
            cls,
            desired_state: StateTypeInner,
            shared_info_setter_option: Mapping[str, Any] | None = None,
            send_state: bool = False,
            prev_timestep_id_dict_option: Any | None = None,
        ) -> EnvActionResponse.SET_STATE[AgentIDInner, StateTypeInner]: ...
