from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

from rlgym.api import AgentID, ObsType

_T = TypeVar("_T")

from torch import Tensor

class FlattenedState: ...

def flatten_env_obs_data_dict(
    env_obs_data_dict: Mapping[
        int,
        tuple[
            Sequence[AgentID],
            Sequence[ObsType],
        ],
    ],
) -> tuple[tuple[list[AgentID], list[ObsType]], FlattenedState]: ...
def unflatten_iterable(
    seq: Iterable[_T], state: FlattenedState
) -> dict[int, list[_T]]: ...
def unflatten_tensor(t: Tensor, state: FlattenedState) -> dict[int, Tensor]: ...
