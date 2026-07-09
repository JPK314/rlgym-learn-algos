from abc import ABC, abstractmethod
from typing import Any, Generic

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn.api import AgentControllerConfig
from typing_extensions import override

from .metrics_logger import AgentControllerData, MetricsLogger, MetricsLoggerConfig


def print_dict(d: dict[Any, Any], indent: str = ""):
    deferred_list: list[tuple[str, Any]] = []
    for k, v in d.items():
        if isinstance(k, str):
            k_str = k
        else:
            k_str = repr(k)
        if isinstance(v, dict):
            deferred_list.append((k_str, v))  # pyright: ignore [reportUnknownArgumentType]
            continue
        if isinstance(v, str):
            v_str = v
        else:
            v_str = repr(v)
        print(f"{indent}{k_str}: {v_str}")
    for k_str, v in deferred_list:
        print(f"{indent}-- {k_str} --")
        print_dict(v, indent + "  ")


class DictMetricsLogger(
    MetricsLogger[
        AgentControllerConfig,
        MetricsLoggerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        AgentControllerData,
    ],
    ABC,
    Generic[
        AgentControllerConfig,
        MetricsLoggerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        AgentControllerData,
    ],
):
    """
    This is a specification of the MetricsLogger which provides an additional method get_metrics to retrieve the metrics as a dictionary.
    """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        :return: metrics data for consumption and side effects by the caller, in the form of a dictionary
        """

    @override
    def report_metrics(self):
        print_dict(self.get_metrics())
