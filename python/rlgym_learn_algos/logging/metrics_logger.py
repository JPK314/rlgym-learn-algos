from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
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
from rlgym_learn.api import (
    AgentControllerConfig,
    DerivedAgentControllerConfig,
)

MetricsLoggerConfig = TypeVar("MetricsLoggerConfig", bound=InstanceOf[BaseModel] | None)
AgentControllerData = TypeVar("AgentControllerData")


@dataclass
class DerivedMetricsLoggerConfig(
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
    ]
):
    controller_name: str | None
    derived_agent_controller_config: DerivedAgentControllerConfig[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
    metrics_logger_config: MetricsLoggerConfig
    checkpoint_load_folder: str | None = None


# TODO: update docs
class MetricsLogger(
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
    This class is designed to be used inside an agent controller to handle the processing of state metrics and agent controller data, and to have some side effects resulting from said processing. It supports config-based saving and loading, and nesting with other MetricsLogger subclasses' config-based saving and loading via the AdditionalDerivedConfig.

    MetricsLoggerConfig is the (pydantic) config model for the class, or None if no config is needed.

    MetricsLoggerAdditionalConfig is a dataclass that can include arbitrary data (usually from other config models). It is the responsibility of the agent controller to instantiate this if it's needed.

    StateMetrics is the type used for collection of data from the environment processes.

    AgentControllerData is the type used for collection of data from the agent controller containing this metrics logger.
    """

    @property
    @abstractmethod
    def config_model(self) -> type[MetricsLoggerConfig] | None:
        """
        Function to return the config model type that your MetricsLogger implementation uses, or None if no config model is used.
        """

    @abstractmethod
    def collect_env_metrics(self, data: list[dict[str, Any] | None]):
        """
        This method is intended to allow batch processing of env metrics using the shared info deserialized from the env processes. The result of processing should be stored and used the next time report_metrics is called.
        There is no guarantee that this method will only be called once between each report_metrics call. The list will only contain Nones if shared_info_serde_type is set to None in SerdeTypesModel.
        """

    @abstractmethod
    def collect_agent_metrics(self, data: AgentControllerData):
        """
        This method is intended to allow processing of AgentControllerData after it gets finalized by the agent controller. The result of processing should be stored and used the next time report_metrics is called.
        There is no guarantee that this method will only be called once between each report_metrics call.
        """

    @abstractmethod
    def report_metrics(self) -> None:
        """
        This method is intended to have arbitrary side effects based on data collected so far. This could be printing, or logging to wandb, or sending data to a redis server, or whatever.
        """

    @abstractmethod
    def load(
        self,
        config: DerivedMetricsLoggerConfig[
            AgentControllerConfig,
            MetricsLoggerConfig,
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
        Sets data inside this instance using config, which may include loading data from a checkpoint.
        """

    @abstractmethod
    def save_checkpoint(self, folder_path: str | PathLike[str]):
        """
        Saves data inside this instance which needs to be checkpointed.
        """
