from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypedDict, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from rlgym_learn import EnvAction, EnvActionType, EnvCloseReason, Timestep
from rlgym_learn.api import AgentController, DerivedAgentControllerConfig
from torch import device as _device
from typing_extensions import override

from rlgym_learn_algos.agent_controller.multi_agent import (
    DerivedMultiAgentSubcontrollerConfig,
)

from ..agent_controller import MultiAgentSubcontroller
from ..logging import (
    DerivedMetricsLoggerConfig,
    MetricsLogger,
    MetricsLoggerConfig,
)
from ..stateful_functions import ObsStandardizer
from ..util import flatten_env_obs_data_dict, unflatten_iterable
from .actor import Actor
from .critic import Critic
from .env_trajectories import EnvTrajectories
from .experience_buffer import (
    DerivedExperienceBufferConfig,
    ExperienceBuffer,
    ExperienceBufferConfigModel,
)
from .ppo_learner import (
    DerivedPPOLearnerConfig,
    PPOData,
    PPOLearner,
    PPOLearnerConfigModel,
)
from .trajectory import Trajectory
from .trajectory_processor import TrajectoryProcessorConfig, TrajectoryProcessorData

EXPERIENCE_BUFFER_FOLDER = "experience_buffer"
PPO_LEARNER_FOLDER = "ppo_learner"
METRICS_LOGGER_FOLDER = "metrics_logger"
PPO_AGENT_FILE = "ppo_agent.json"
ITERATION_TRAJECTORIES_FILE = "iteration_trajectories.pkl"
ITERATION_SHARED_INFOS_FILE = "iteration_shared_infos.pkl"


class PPOAgentControllerConfigModel(
    BaseModel, Generic[TrajectoryProcessorConfig, MetricsLoggerConfig], extra="forbid"
):
    timesteps_per_iteration: int = 50000
    save_every_ts: int = 1_000_000
    run_suffix: str = Field(default_factory=lambda: f"-{time.time_ns()}")
    checkpoint_load_folder: str | None = None
    n_checkpoints_to_keep: int = 5
    random_seed: int = 123
    save_mid_iteration_data_in_checkpoint: bool = True
    learner_config: PPOLearnerConfigModel = Field(
        default_factory=lambda: PPOLearnerConfigModel()
    )
    experience_buffer_config: ExperienceBufferConfigModel[TrajectoryProcessorConfig] = (
        Field(default_factory=lambda: ExperienceBufferConfigModel())  # pyright: ignore [reportAssignmentType]
    )
    run_name: str = "rlgym-learn-run"
    metrics_logger_config: MetricsLoggerConfig = None  # pyright: ignore [reportAssignmentType]

    @model_validator(mode="before")
    @classmethod
    def validate_metrics_logger_and_experience_buffer_config_models(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        ppo_agent_controller: (
            PPOAgentController[
                TrajectoryProcessorConfig,
                MetricsLoggerConfig,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
            ]
            | None
        ) = info.context
        data_dict = data
        if ppo_agent_controller is not None and isinstance(data_dict, dict):
            data_dict = cast(dict[Any, Any], data_dict)
            if (
                ppo_agent_controller.metrics_logger is not None
                and "metrics_logger_config" in data_dict
            ):
                metrics_logger_config_raw = data_dict["metrics_logger_config"]
                if isinstance(metrics_logger_config_raw, dict):
                    metrics_logger_config_model_type = (
                        ppo_agent_controller.metrics_logger.config_model
                    )
                    if metrics_logger_config_model_type is None:
                        metrics_logger_config = None
                    else:
                        metrics_logger_config = cast(
                            BaseModel, metrics_logger_config_model_type
                        ).model_validate(
                            metrics_logger_config_raw,
                            context=ppo_agent_controller.metrics_logger,
                        )
                else:
                    metrics_logger_config = metrics_logger_config_raw
                data_dict["metrics_logger_config"] = metrics_logger_config
            if "experience_buffer_config" in data_dict:
                experience_buffer_config_raw = data_dict["experience_buffer_config"]
                if isinstance(experience_buffer_config_raw, dict):
                    experience_buffer_config = ExperienceBufferConfigModel[
                        TrajectoryProcessorConfig
                    ].model_validate(
                        experience_buffer_config_raw,
                        context=ppo_agent_controller.experience_buffer,
                    )
                else:
                    experience_buffer_config = experience_buffer_config_raw
                data_dict["experience_buffer_config"] = experience_buffer_config
        return data


@dataclass
class PPOAgentControllerData(Generic[TrajectoryProcessorData]):
    ppo_data: PPOData
    trajectory_processor_data: TrajectoryProcessorData
    cumulative_timesteps: int
    iteration_time: float
    timesteps_collected: int
    timestep_collection_time: float


class PPOAgentStateDict(TypedDict):
    cur_iteration: int
    iteration_timesteps: int
    cumulative_timesteps: int
    iteration_start_time: float
    timestep_collection_start_time: float


class PPOAgentController(
    AgentController[
        PPOAgentControllerConfigModel[TrajectoryProcessorConfig, MetricsLoggerConfig],
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    MultiAgentSubcontroller[
        PPOAgentControllerConfigModel[TrajectoryProcessorConfig, MetricsLoggerConfig],
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    Generic[
        TrajectoryProcessorConfig,
        MetricsLoggerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        TrajectoryProcessorData,
    ],
):
    def __init__(
        self,
        actor_factory: Callable[
            [ObsSpaceType, ActionSpaceType, _device],
            Actor[AgentID, ObsType, ActionType],
        ],
        critic_factory: Callable[[ObsSpaceType, _device], Critic[AgentID, ObsType]],
        experience_buffer: ExperienceBuffer[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ],
        metrics_logger: MetricsLogger[
            PPOAgentControllerConfigModel[
                TrajectoryProcessorConfig, MetricsLoggerConfig
            ],
            MetricsLoggerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
            PPOAgentControllerData[TrajectoryProcessorData],
        ]
        | None = None,
        obs_standardizer: ObsStandardizer[AgentID, ObsType] | None = None,
    ):
        self.learner: PPOLearner[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            ObsSpaceType,
            ActionSpaceType,
            TrajectoryProcessorData,
        ] = PPOLearner(actor_factory, critic_factory)
        self.experience_buffer: ExperienceBuffer[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ] = experience_buffer
        self.metrics_logger: (
            MetricsLogger[
                PPOAgentControllerConfigModel[
                    TrajectoryProcessorConfig, MetricsLoggerConfig
                ],
                MetricsLoggerConfig,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                PPOAgentControllerData[TrajectoryProcessorData],
            ]
            | None
        ) = metrics_logger
        self.obs_standardizer: ObsStandardizer[AgentID, ObsType] | None = (
            obs_standardizer
        )
        if obs_standardizer is not None:
            print(
                "Warning: using an obs standardizer is slow! It is recommended to design your obs to be standardized (i.e. have approximately mean 0 and std 1 for each value) without needing this extra post-processing step."
            )

        self.current_env_trajectories: dict[
            int,
            EnvTrajectories[AgentID, ObsType, ActionType, RewardType],
        ] = {}
        self.current_env_controlled_agent_ids: dict[int, list[AgentID]] = {}
        self.current_env_log_probs: dict[int, list[np.ndarray]] = {}
        self.iteration_trajectories: list[
            Trajectory[AgentID, ObsType, ActionType, RewardType]
        ] = []
        self.iteration_shared_infos: list[dict[str, Any] | None] = []
        self.cur_iteration: int = 0
        self.iteration_timesteps: int = 0
        self.cumulative_timesteps: int = 0
        cur_time = time.perf_counter()
        self.iteration_start_time: float = cur_time
        self.timestep_collection_start_time: float = cur_time
        self.timestep_collection_end_time: float
        self.ts_since_last_save: int = 0
        self.set_spaces: bool = False
        self.obs_space: ObsSpaceType
        self.action_space: ActionSpaceType
        self.config: DerivedMultiAgentSubcontrollerConfig[
            PPOAgentControllerConfigModel[
                TrajectoryProcessorConfig, MetricsLoggerConfig
            ],
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ]
        self.checkpoints_save_folder: str

    @property
    @override
    def config_model(self):
        return PPOAgentControllerConfigModel

    @override
    def set_space_types(
        self,
        env_spaces_data_dict: dict[
            int, dict[AgentID, tuple[ObsSpaceType, ActionSpaceType]]
        ],
    ):
        obs_space: ObsSpaceType | None = None
        action_space: ActionSpaceType | None = None
        if self.set_spaces:
            obs_space = self.obs_space
            action_space = self.action_space
        for spaces_data_dict in env_spaces_data_dict.values():
            for agent_obs_space, agent_action_space in spaces_data_dict.values():
                if obs_space is not None:
                    assert obs_space == agent_obs_space, (
                        "Please create a subclass of PPOAgentController and override the set_space_types method if the environment can return more than one distinct ObsSpaceType value"
                    )
                obs_space = agent_obs_space
                if action_space is not None:
                    assert action_space == agent_action_space, (
                        "Please create a subclass of PPOAgentController and override the set_space_types method if the environment can return more than one distinct ActionSpaceType value"
                    )
                action_space = agent_action_space
        assert obs_space is not None, (
            "self.obs_space could not be determined after set_space_types was called"
        )
        assert action_space is not None, (
            "self.action_space could not be determined after set_space_types was called"
        )
        self.obs_space = obs_space
        self.action_space = action_space
        self.set_spaces = True

    @override
    def load(
        self,
        config: DerivedAgentControllerConfig[
            PPOAgentControllerConfigModel[
                TrajectoryProcessorConfig, MetricsLoggerConfig
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
        self.subcontroller_load(
            DerivedMultiAgentSubcontrollerConfig[
                PPOAgentControllerConfigModel[
                    TrajectoryProcessorConfig, MetricsLoggerConfig
                ],
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ].from_agent_controller_config(config, "PPOController")
        )

    @override
    def subcontroller_load(
        self,
        config: DerivedMultiAgentSubcontrollerConfig[
            PPOAgentControllerConfigModel[
                TrajectoryProcessorConfig, MetricsLoggerConfig
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
        self.config = config
        assert not self.config.process_config.recalculate_agent_id_every_step, (
            f"{self.config.subcontroller_name}: PPO Agent Controller cannot handle agent ids being recalculated every step!"
        )
        print(
            f"{self.config.subcontroller_name}: Using device {config.subcontroller_config.learner_config.device}"
        )
        agent_controller_config = config.subcontroller_config
        learner_config = config.subcontroller_config.learner_config
        experience_buffer_config = config.subcontroller_config.experience_buffer_config
        learner_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, PPO_LEARNER_FOLDER
            )
        )
        experience_buffer_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, EXPERIENCE_BUFFER_FOLDER
            )
        )
        metrics_logger_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, METRICS_LOGGER_FOLDER
            )
        )

        if agent_controller_config.checkpoint_load_folder is not None:
            loaded_checkpoint_runs_folder = os.path.abspath(
                os.path.join(agent_controller_config.checkpoint_load_folder, "../..")
            )
            abs_save_folder = os.path.abspath(config.save_folder)
            # TODO: this doesn't seem to be working
            if abs_save_folder == loaded_checkpoint_runs_folder:
                print(
                    f"{self.config.subcontroller_name}: Using the loaded checkpoint's run folder as the checkpoints save folder."
                )
                checkpoints_save_folder = os.path.abspath(
                    os.path.join(agent_controller_config.checkpoint_load_folder, "..")
                )
            else:
                print(
                    f"{self.config.subcontroller_name}: Runs folder in config does not align with loaded checkpoint's runs folder. Creating new run in the config-based runs folder."
                )
                checkpoints_save_folder = os.path.join(
                    config.save_folder,
                    agent_controller_config.run_name
                    + agent_controller_config.run_suffix,
                )
        else:
            checkpoints_save_folder = os.path.join(
                config.save_folder,
                agent_controller_config.run_name + agent_controller_config.run_suffix,
            )
        self.checkpoints_save_folder = checkpoints_save_folder
        print(
            f"{self.config.subcontroller_name}: Saving checkpoints to {self.checkpoints_save_folder}"
        )

        self.learner.load(
            DerivedPPOLearnerConfig(
                learner_config=learner_config,
                agent_controller_name=self.config.subcontroller_name,
                obs_space=self.obs_space,
                action_space=self.action_space,
                checkpoint_load_folder=learner_checkpoint_load_folder,
            )
        )
        self.experience_buffer.load(
            DerivedExperienceBufferConfig(
                experience_buffer_config=experience_buffer_config,
                agent_controller_name=self.config.subcontroller_name,
                seed=config.base_config.random_seed,
                dtype=agent_controller_config.learner_config.dtype,
                learner_device=agent_controller_config.learner_config.device,
                checkpoint_load_folder=experience_buffer_checkpoint_load_folder,
            )
        )
        if self.metrics_logger is not None:
            self.metrics_logger.load(
                DerivedMetricsLoggerConfig(
                    controller_name=config.subcontroller_name,
                    derived_agent_controller_config=config.to_agent_controller_config(),
                    metrics_logger_config=self.config.subcontroller_config.metrics_logger_config,
                    checkpoint_load_folder=metrics_logger_checkpoint_load_folder,
                )
            )

        if agent_controller_config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

        _ = torch.manual_seed(self.config.base_config.random_seed)  # pyright: ignore [reportUnknownMemberType]
        np.random.seed(self.config.base_config.random_seed)
        random.seed(self.config.base_config.random_seed)

    def _load_from_checkpoint(self):
        assert self.config.subcontroller_config.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint when no checkpoint load folder is in config!"
        )
        try:
            with open(
                os.path.join(
                    self.config.subcontroller_config.checkpoint_load_folder,
                    ITERATION_TRAJECTORIES_FILE,
                ),
                "rb",
            ) as f:
                iteration_trajectories: list[
                    Trajectory[AgentID, ObsType, ActionType, RewardType]
                ] = pickle.load(f)
        except FileNotFoundError:
            print(
                f"{self.config.subcontroller_name}: Tried to load current trajectories from checkpoint using the file at location {os.path.join(self.config.subcontroller_config.checkpoint_load_folder, ITERATION_TRAJECTORIES_FILE)}, but there is no such file! Current trajectories will be initialized as an empty list instead."
            )
            iteration_trajectories = []
        try:
            with open(
                os.path.join(
                    self.config.subcontroller_config.checkpoint_load_folder,
                    ITERATION_SHARED_INFOS_FILE,
                ),
                "rb",
            ) as f:
                iteration_shared_infos: list[dict[str, Any] | None] = pickle.load(f)
        except FileNotFoundError:
            print(
                f"{self.config.subcontroller_name}: Tried to load iteration shared info data from checkpoint using the file at location {os.path.join(self.config.subcontroller_config.checkpoint_load_folder, ITERATION_SHARED_INFOS_FILE)}, but there is no such file! Iteration shared info data will be initialized as an empty list instead."
            )
            iteration_shared_infos = []
        try:
            with open(
                os.path.join(
                    self.config.subcontroller_config.checkpoint_load_folder,
                    PPO_AGENT_FILE,
                ),
                "rt",
            ) as f:
                state: PPOAgentStateDict = json.load(f)
        except FileNotFoundError:
            print(
                f"{self.config.subcontroller_name}: Tried to load PPO agent miscellaneous state data from checkpoint using the file at location {os.path.join(self.config.subcontroller_config.checkpoint_load_folder, PPO_AGENT_FILE)}, but there is no such file! This state data will be initialized as if this were a new run instead."
            )
            state = {
                "cur_iteration": 0,
                "iteration_timesteps": 0,
                "cumulative_timesteps": 0,
                "iteration_start_time": time.perf_counter(),
                "timestep_collection_start_time": time.perf_counter(),
            }

        self.iteration_trajectories = iteration_trajectories
        self.iteration_shared_infos = iteration_shared_infos
        self.cur_iteration = state["cur_iteration"]
        self.iteration_timesteps = state["iteration_timesteps"]
        self.cumulative_timesteps = state["cumulative_timesteps"]
        # I'm aware that loading these start times will cause some funny numbers for the first iteration
        self.iteration_start_time = state["iteration_start_time"]
        self.timestep_collection_start_time = state["timestep_collection_start_time"]

    @override
    def save_checkpoint(self):
        print(f"Saving checkpoint {self.cumulative_timesteps}...")

        checkpoint_save_folder = os.path.join(
            self.checkpoints_save_folder, str(time.time_ns())
        )
        os.makedirs(checkpoint_save_folder, exist_ok=True)
        self.learner.save_checkpoint(
            os.path.join(checkpoint_save_folder, PPO_LEARNER_FOLDER)
        )
        self.experience_buffer.save_checkpoint(
            os.path.join(checkpoint_save_folder, EXPERIENCE_BUFFER_FOLDER)
        )
        if self.metrics_logger is not None:
            self.metrics_logger.save_checkpoint(
                os.path.join(checkpoint_save_folder, METRICS_LOGGER_FOLDER)
            )

        if self.config.subcontroller_config.save_mid_iteration_data_in_checkpoint:
            with open(
                os.path.join(checkpoint_save_folder, ITERATION_TRAJECTORIES_FILE),
                "wb",
            ) as f:
                pickle.dump(self.iteration_trajectories, f)
            with open(
                os.path.join(checkpoint_save_folder, ITERATION_SHARED_INFOS_FILE),
                "wb",
            ) as f:
                pickle.dump(self.iteration_shared_infos, f)
        with open(os.path.join(checkpoint_save_folder, PPO_AGENT_FILE), "wt") as f:
            state = {
                "cur_iteration": self.cur_iteration,
                "iteration_timesteps": self.iteration_timesteps,
                "cumulative_timesteps": self.cumulative_timesteps,
                "iteration_start_time": self.iteration_start_time,
                "timestep_collection_start_time": self.timestep_collection_start_time,
            }
            json.dump(state, f, indent=4)

        # TODO: does this actually work? I'm not sure the file structure I'm using actually works with this assumption
        # Prune old checkpoints
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
        ]
        if (
            len(existing_checkpoints)
            > self.config.subcontroller_config.n_checkpoints_to_keep
        ):
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[
                : -self.config.subcontroller_config.n_checkpoints_to_keep
            ]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

    @torch.no_grad
    @override
    def get_actions(
        self, env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]]
    ) -> Mapping[int, Iterable[ActionType]]:
        if self.config.subcontroller_mode:
            self.current_env_controlled_agent_ids.update(
                {
                    env_id: agent_id_list
                    for (
                        env_id,
                        (agent_id_list, _),
                    ) in env_obs_data_dict.items()
                }
            )

        ((agent_id_list, obs_list), flattened_state) = flatten_env_obs_data_dict(
            env_obs_data_dict
        )

        actions, log_probs = self.learner.actor.get_action(agent_id_list, obs_list)

        if log_probs.dim() == 0:
            # This can happen if the input is a single element
            log_probs = log_probs.unsqueeze(0)

        env_action_dict = unflatten_iterable(actions, flattened_state)
        self.current_env_log_probs.update(
            unflatten_iterable(log_probs.cpu().numpy(), flattened_state)
        )
        return env_action_dict

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
        env_actions: dict[int, EnvAction[AgentID, ActionType, StateType]] = {}
        step_env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]] = {}
        for env_id, obs_data in env_obs_data_dict.items():
            if env_id not in self.current_env_trajectories:
                # This must be the first env action after a reset, so we step
                step_env_obs_data_dict[env_id] = obs_data
                continue
            done = all(self.current_env_trajectories[env_id].dones.values())
            if done:
                env_actions[env_id] = EnvAction.RESET()
            else:
                step_env_obs_data_dict[env_id] = obs_data

        if step_env_obs_data_dict:
            env_actions.update(
                {
                    env_id: EnvAction.STEP(action_list=action_list)
                    for env_id, action_list in self.get_actions(
                        step_env_obs_data_dict
                    ).items()
                }
            )
        self.process_env_actions(env_actions)
        return (0, env_actions)

    def _standardize_timestep_observations(
        self,
        timesteps: list[Timestep[AgentID, ObsType, ActionType, RewardType]],
    ):
        if self.obs_standardizer is None:
            return
        agent_id_list: list[AgentID | None] = [None] * (2 * len(timesteps))
        obs_list: list[ObsType | None] = [None] * len(agent_id_list)
        for timestep_idx, timestep in enumerate(timesteps):
            agent_id_list[2 * timestep_idx] = timestep.agent_id
            agent_id_list[2 * timestep_idx + 1] = timestep.agent_id
            obs_list[2 * timestep_idx] = timestep.obs
            obs_list[2 * timestep_idx + 1] = timestep.next_obs
        standardized_obs = self.obs_standardizer.standardize(
            cast(list[AgentID], agent_id_list), cast(list[ObsType], obs_list)
        )
        for obs_idx, obs in enumerate(standardized_obs):
            if obs_idx % 2 == 0:
                timesteps[obs_idx // 2].obs = obs
            else:
                timesteps[obs_idx // 2].next_obs = obs

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
        timesteps_added = 0
        shared_infos: list[dict[str, Any] | None] = []
        for env_id, (
            env_timesteps,
            env_shared_info,
            _,
        ) in timestep_data.items():
            if self.obs_standardizer is not None:
                self._standardize_timestep_observations(env_timesteps)
            if env_timesteps:
                if env_id not in self.current_env_trajectories:
                    self.current_env_trajectories[env_id] = EnvTrajectories(
                        [timestep.agent_id for timestep in env_timesteps]
                    )
                timesteps_added += self.current_env_trajectories[env_id].add_steps(
                    None
                    if not self.config.subcontroller_mode
                    else self.current_env_controlled_agent_ids[env_id],
                    env_timesteps,
                    self.current_env_log_probs[env_id],
                )
            shared_infos.append(env_shared_info)
        self.iteration_timesteps += timesteps_added
        self.cumulative_timesteps += timesteps_added
        self.iteration_shared_infos += shared_infos
        if (
            self.iteration_timesteps
            >= self.config.subcontroller_config.timesteps_per_iteration
        ):
            self.timestep_collection_end_time = time.perf_counter()
            self._learn()
            self.cur_iteration += 1
        if self.ts_since_last_save >= self.config.subcontroller_config.save_every_ts:
            self.save_checkpoint()
            self.ts_since_last_save = 0

    @override
    def process_env_actions(
        self, env_actions: dict[int, EnvAction[AgentID, ActionType, StateType]]
    ):
        for env_id, env_action in env_actions.items():
            # this is a getter so we only want to do it once
            enum_type = env_action.enum_type
            if enum_type == EnvActionType.STEP:
                pass
            elif enum_type == EnvActionType.RESET:
                _ = self.current_env_controlled_agent_ids.pop(env_id, None)
                env_trajectories = self.current_env_trajectories.pop(env_id)
                env_trajectories.finalize()
                self.iteration_trajectories += env_trajectories.get_trajectories()
            elif enum_type == EnvActionType.SET_STATE:
                # Can get the desired_state using env_action.desired_state and the prev_timestep_id_dict using env_action.prev_timestep_id_dict, but I'll leave that to you
                raise NotImplementedError
            else:
                raise ValueError

    def _learn(self):
        env_trajectories_list = list(self.current_env_trajectories.values())
        for env_trajectories in env_trajectories_list:
            env_trajectories.finalize()
            self.iteration_trajectories += env_trajectories.get_trajectories()
        self._update_value_predictions()
        trajectory_processor_data = self.experience_buffer.submit_experience(
            self.iteration_trajectories
        )
        ppo_data = self.learner.learn(self.experience_buffer)

        cur_time = time.perf_counter()
        if self.metrics_logger is not None:
            self.metrics_logger.collect_agent_metrics(
                PPOAgentControllerData(
                    ppo_data,
                    trajectory_processor_data,
                    self.cumulative_timesteps,
                    cur_time - self.iteration_start_time,
                    self.iteration_timesteps,
                    self.timestep_collection_end_time
                    - self.timestep_collection_start_time,
                )
            )
            self.metrics_logger.collect_env_metrics(self.iteration_shared_infos)
            self.metrics_logger.report_metrics()

        self.iteration_trajectories.clear()
        self.iteration_shared_infos.clear()
        self.current_env_trajectories.clear()
        self.ts_since_last_save += self.iteration_timesteps
        self.iteration_timesteps = 0
        self.iteration_start_time = cur_time
        self.timestep_collection_start_time = time.perf_counter()

    @torch.no_grad()
    def _update_value_predictions(self):
        """
        Function to update the value predictions inside the Trajectory instances of self.iteration_trajectories
        """
        traj_timestep_idx_ranges: list[tuple[int, int]] = []
        start = 0
        stop = 0
        critic_agent_id_input: list[AgentID] = []
        critic_obs_input: list[ObsType] = []
        for trajectory in self.iteration_trajectories:
            obs_list = trajectory.obs_list + [cast(ObsType, trajectory.final_obs)]
            traj_len = len(obs_list)
            agent_id_list = [trajectory.agent_id] * traj_len
            stop = start + traj_len
            critic_agent_id_input += agent_id_list
            critic_obs_input += obs_list
            traj_timestep_idx_ranges.append((start, stop))
            start = stop

        val_preds: torch.Tensor = (
            self.learner.critic(critic_agent_id_input, critic_obs_input)
            .flatten()
            .to(device="cpu", non_blocking=True)
        )
        torch.cuda.empty_cache()
        for idx, (start, stop) in enumerate(traj_timestep_idx_ranges):
            self.iteration_trajectories[idx].val_preds = val_preds[start : stop - 1]
            self.iteration_trajectories[idx].final_val_pred = val_preds[stop - 1]
        if self.config.subcontroller_config.learner_config.device.type != "cpu":
            torch.cuda.current_stream().synchronize()

    @override
    def handle_env_closes(self, env_close_reason_dict: dict[int, EnvCloseReason]):
        # Nothing to do here, all the dicts using env ids as keys get those keys cleared out upon the next _learn call
        pass

    @override
    def cleanup(self):
        # Nothing to do here
        pass
