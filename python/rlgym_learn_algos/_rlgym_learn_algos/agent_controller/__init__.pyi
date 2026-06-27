# pyright: reportUnusedParameter=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, final

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
from rlgym_learn.api import AgentController

@final
class MultiAgentControllerManager(
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
        agent_controllers: Sequence[
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
        ],
    ) -> MultiAgentControllerManager[
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
            str,
            tuple[
                Sequence[AgentID],
                Sequence[ObsType],
            ],
        ],
        state_info: Mapping[
            str,
            tuple[
                Mapping[str, Any] | None,
                StateType | None,
                Mapping[AgentID, bool] | None,
                Mapping[AgentID, bool] | None,
            ],
        ],
    ) -> dict[str, EnvAction]: ...
