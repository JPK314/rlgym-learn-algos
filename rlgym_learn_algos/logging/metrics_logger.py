from abc import abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type
from pydantic import BaseModel

from rlgym_learn.api import AgentControllerData

MetricsLoggerConfig = TypeVar("MetricsLoggerConfig", bound=Optional[BaseModel])
MetricsLoggerAdditionalDerivedConfig = TypeVar("MetricsLoggerAdditionalDerivedConfig")


@dataclass
class DerivedMetricsLoggerConfig(
    Generic[MetricsLoggerConfig, MetricsLoggerAdditionalDerivedConfig]
):
    metrics_logger_config: MetricsLoggerConfig = None
    checkpoint_load_folder: Optional[str] = None
    agent_controller_name: str = ""
    additional_derived_config: MetricsLoggerAdditionalDerivedConfig = None


# TODO: docs
class MetricsLogger(
    Generic[
        MetricsLoggerConfig,
        MetricsLoggerAdditionalDerivedConfig,
        AgentControllerData,
    ]
):
    """
    This class is designed to be used inside an agent controller to handle the processing of state metrics and agent controller data, and to have some side effects resulting from said processing. It supports config-based saving and loading, and nesting with other MetricsLogger subclasses' config-based saving and loading via the AdditionalDerivedConfig.

    MetricsLoggerConfig is the (pydantic) config model for the class, or None if no config is needed.

    MetricsLoggerAdditionalConfig is a dataclass that can include arbitrary data (usually from other config models). It is the responsibility of the agent controller to instantiate this if it's needed.

    StateMetrics is the type used for collection of data from the environment processes.

    AgentControllerData is the type used for collection of data from the agent controller containing this metrics logger.
    """

    @property
    def config_model(self) -> Type[MetricsLoggerConfig]:
        """
        Function to return the config model type that your MetricsLogger implementation uses. Defaults to NoneType.
        """
        return type(None)

    def collect_env_metrics(self, data: List[Dict[str, Any]]):
        """
        This method is intended to allow batch processing of env metrics using the shared info deserialized from the env processes. The result of processing should be stored and used the next time report_metrics is called.
        There is no guarantee that this method will only be called once between each report_metrics call.
        """
        pass

    def collect_agent_metrics(self, data: AgentControllerData):
        """
        This method is intended to allow processing of AgentControllerData after it gets finalized by the agent controller. The result of processing should be stored and used the next time report_metrics is called.
        There is no guarantee that this method will only be called once between each report_metrics call.
        """
        pass

    @abstractmethod
    def report_metrics(self):
        """
        This method is intended to have arbitrary side effects based on data collected so far. This could be printing, or logging to wandb, or sending data to a redis server, or whatever.
        """
        raise NotImplementedError

    def load(
        self,
        config: DerivedMetricsLoggerConfig[
            MetricsLoggerConfig, MetricsLoggerAdditionalDerivedConfig
        ],
    ):
        """
        Sets data inside this instance using config, which may include loading data from a checkpoint.
        """
        pass

    def save_checkpoint(self, folder_path: str | PathLike[str]):
        """
        Saves data inside this instance which needs to be checkpointed.
        """
        pass
