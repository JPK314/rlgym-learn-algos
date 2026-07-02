from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from rlgym.api import AgentID, ObsType

T = TypeVar("T")

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor: TypeAlias = Any

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
    seq: Iterable[T], state: FlattenedState
) -> dict[int, list[T]]: ...
def unflatten_tensor(t: Tensor, state: FlattenedState) -> dict[int, Tensor]: ...
