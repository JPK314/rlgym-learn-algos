import json
import os
import time
from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Generic, TypeVar, cast

import wandb
from pydantic import BaseModel, Field, InstanceOf, ValidationInfo, model_validator
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
from typing_extensions import override

from ..dict_metrics_logger import DictMetricsLogger
from ..metrics_logger import (
    AgentControllerData,
    DerivedMetricsLoggerConfig,
    MetricsLogger,
)

# wandb can create a /wandb folder on sys.path that python thinks is a module it can import.
# If wandb gets uninstalled but this folder stays then python will resolve this folder as a module it can import, which causes confusion
if wandb.__file__ is None:  # pyright: ignore [reportUnnecessaryComparison]
    raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

InnerMetricsLoggerConfig = TypeVar(
    "InnerMetricsLoggerConfig", bound=InstanceOf[BaseModel] | None
)


def convert_nested_dict(d: dict[str, Any]):
    new: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            converted = convert_nested_dict(
                {str(k1): v1 for (k1, v1) in cast(dict[Any, Any], v).items()}
            )
            to_add = {f"{k}/{k1}": v1 for k1, v1 in converted.items()}
        else:
            to_add = {k: v}
        new = {**new, **to_add}
    return new


class WandbMetricsLoggerConfigModel(
    BaseModel, Generic[InnerMetricsLoggerConfig], extra="forbid"
):
    inner_metrics_logger_config: InnerMetricsLoggerConfig
    enable: bool = True
    project: str = "rlgym-learn"
    group: str = "unnamed-runs"
    run: str = "rlgym-learn-run"
    id: str | None = None
    new_run_with_run_suffix: bool = False
    additional_wandb_run_config: dict[str, Any] = Field(default_factory=dict)
    settings_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_metrics_logger_config_model(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        wandb_metrics_logger: (
            WandbMetricsLogger[
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                InnerMetricsLoggerConfig,
            ]
            | None
        ) = info.context
        data_dict = data
        if (
            wandb_metrics_logger is not None
            and isinstance(data_dict, dict)
            and "inner_metrics_logger_config" in data_dict
        ):
            data_dict = cast(dict[Any, Any], data_dict)
            inner_metrics_logger_config_raw = data_dict["inner_metrics_logger_config"]
            if isinstance(inner_metrics_logger_config_raw, dict):
                inner_metrics_logger_config_model_type = (
                    wandb_metrics_logger.inner_metrics_logger.config_model
                )
                if inner_metrics_logger_config_model_type is None:
                    inner_metrics_logger_config = None
                else:
                    inner_metrics_logger_config = cast(
                        BaseModel, inner_metrics_logger_config_model_type
                    ).model_validate(
                        inner_metrics_logger_config_raw,
                        context=wandb_metrics_logger.inner_metrics_logger,
                    )
            else:
                inner_metrics_logger_config = inner_metrics_logger_config_raw
            data_dict["inner_metrics_logger_config"] = inner_metrics_logger_config
        return data


@dataclass
class WandbAdditionalDerivedConfig:
    derived_wandb_run_config: dict[str, Any] = Field(default_factory=dict)
    run_suffix: str | None = None


class WandbMetricsLogger(
    MetricsLogger[
        AgentControllerConfig,
        WandbMetricsLoggerConfigModel[InnerMetricsLoggerConfig],
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        AgentControllerData,
    ],
    Generic[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        InnerMetricsLoggerConfig,
        AgentControllerData,
    ],
):
    def __init__(
        self,
        inner_metrics_logger: DictMetricsLogger[
            AgentControllerConfig,
            InnerMetricsLoggerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
            AgentControllerData,
        ],
        additional_derived_config_factory: Callable[
            [
                DerivedAgentControllerConfig[
                    AgentControllerConfig,
                    AgentID,
                    ObsType,
                    ActionType,
                    RewardType,
                    StateType,
                    ObsSpaceType,
                    ActionSpaceType,
                ]
            ],
            WandbAdditionalDerivedConfig,
        ]
        | None = None,
        checkpoint_file_name: str = "wandb_metrics_logger.json",
    ):
        self.inner_metrics_logger: DictMetricsLogger[
            AgentControllerConfig,
            InnerMetricsLoggerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
            AgentControllerData,
        ] = inner_metrics_logger
        self.additional_derived_config_factory: (
            Callable[
                [
                    DerivedAgentControllerConfig[
                        AgentControllerConfig,
                        AgentID,
                        ObsType,
                        ActionType,
                        RewardType,
                        StateType,
                        ObsSpaceType,
                        ActionSpaceType,
                    ]
                ],
                WandbAdditionalDerivedConfig,
            ]
            | None
        ) = additional_derived_config_factory
        self.checkpoint_file_name: str = checkpoint_file_name
        self.run_id: str | None = None
        self.wandb_run: wandb.Run | None = None
        self.config: (
            DerivedMetricsLoggerConfig[
                AgentControllerConfig,
                WandbMetricsLoggerConfigModel[InnerMetricsLoggerConfig],
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
            | None
        ) = None
        self.additional_derived_config: WandbAdditionalDerivedConfig | None = None

    @property
    @override
    def config_model(self):
        return WandbMetricsLoggerConfigModel

    @override
    def collect_env_metrics(self, data: list[dict[str, Any] | None]):
        self.inner_metrics_logger.collect_env_metrics(data)

    @override
    def collect_agent_metrics(self, data: AgentControllerData):
        self.inner_metrics_logger.collect_agent_metrics(data)

    @override
    def report_metrics(self):
        if self.wandb_run is not None:
            self.wandb_run.log(
                convert_nested_dict(self.inner_metrics_logger.get_metrics())
            )
        self.inner_metrics_logger.report_metrics()

    @override
    def load(
        self,
        config: DerivedMetricsLoggerConfig[
            AgentControllerConfig,
            WandbMetricsLoggerConfigModel[InnerMetricsLoggerConfig],
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ):
        self.config = config
        self.additional_derived_config = (
            WandbAdditionalDerivedConfig()
            if self.additional_derived_config_factory is None
            else self.additional_derived_config_factory(
                config.derived_agent_controller_config
            )
        )
        self.inner_metrics_logger.load(
            DerivedMetricsLoggerConfig(
                controller_name=config.controller_name,
                derived_agent_controller_config=config.derived_agent_controller_config,
                metrics_logger_config=config.metrics_logger_config.inner_metrics_logger_config,
                checkpoint_load_folder=config.checkpoint_load_folder,
            )
        )
        if self.config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        if not self.config.metrics_logger_config.enable:
            self.wandb_run = None
            self.run_id = None
            return

        if self.run_id is not None and self.config.metrics_logger_config.id is not None:
            print(
                f"{config.controller_name}: Wandb run id from checkpoint ({self.run_id}) is being overridden by wandb run id from config: {config.metrics_logger_config.id}"
            )
            self.run_id = config.metrics_logger_config.id

        wandb_config = {
            **self.additional_derived_config.derived_wandb_run_config,
            **config.metrics_logger_config.additional_wandb_run_config,
        }

        run_name = config.metrics_logger_config.run
        if config.metrics_logger_config.new_run_with_run_suffix:
            print(
                f"{config.controller_name}: Due to config, a new wandb run is being created with run suffix. This run will use the project and group specified in config, and will use the run name in config prepended to the run suffix."
            )
            if (
                self.additional_derived_config.run_suffix is not None
                and len(self.additional_derived_config.run_suffix) > 0
            ):
                run_name += self.additional_derived_config.run_suffix
            else:
                run_name += f"-{time.time_ns()}"

        self.wandb_run = wandb.init(
            project=config.metrics_logger_config.project,
            group=config.metrics_logger_config.group,
            config=wandb_config,
            name=run_name,
            id=self.run_id,
            resume="allow",
            reinit="create_new",
            settings=wandb.Settings(**config.metrics_logger_config.settings_kwargs),
        )
        self.run_id = self.wandb_run.id
        print(f"{config.controller_name}: Created wandb run! {self.run_id}")

    def _load_from_checkpoint(self):
        assert self.config is not None, (
            "Cannot load from checkpoint before calling load with config!"
        )
        assert self.config.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint if checkpoint load folder is None!"
        )
        try:
            with open(
                os.path.join(
                    self.config.checkpoint_load_folder,
                    self.checkpoint_file_name,
                ),
                "rt",
            ) as f:
                state = json.load(f)
            if "run_id" in state:
                self.run_id = state["run_id"]
            else:
                self.run_id = None
        except FileNotFoundError:
            print(
                f"{self.config.controller_name}: Tried to load wandb run from checkpoint using the file at location {str(os.path.join(self.config.checkpoint_load_folder, self.checkpoint_file_name))}, but there is no such file! A new run will be created based on the config values instead."
            )
            self.run_id = None

    @override
    def save_checkpoint(self, folder_path: str | PathLike[str]):
        os.makedirs(folder_path, exist_ok=True)
        state = {"run_id": self.run_id}
        with open(
            os.path.join(
                folder_path,
                self.checkpoint_file_name,
            ),
            "wt",
        ) as f:
            json.dump(state, f, indent=4)
