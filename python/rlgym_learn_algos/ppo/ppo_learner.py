import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, JsonValue, model_validator
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import nn

from rlgym_learn_algos.util.torch_pydantic import (
    PydanticTorchDevice,
    PydanticTorchDtype,
)

from .actor_critic import ActorCritic
from .experience_buffer import ExperienceBuffer
from .trajectory_processor import TrajectoryProcessorConfig, TrajectoryProcessorData


class PPOLearnerConfigModel(BaseModel, extra="forbid"):
    dtype: PydanticTorchDtype = torch.float32
    n_epochs: int = 1
    batch_size: int = 50000
    n_minibatches: int = 1
    ent_coef: float = 0.005
    clip_range: float = 0.2
    max_grad_norm: float | None = 0.5
    optimizer_named_parameter_group_kwargs: dict[str, dict[str, JsonValue]] = Field(
        default={"actor": {"lr": 3e-4}, "critic": {"lr": 3e-4}}
    )
    advantage_standardization: bool = True
    device: PydanticTorchDevice = Field(default="cpu", validate_default=True)
    cudnn_benchmark_mode: bool = True

    @model_validator(mode="after")
    def validate_cudnn_benchmark(self):
        if self.device.type != "cuda":
            self.cudnn_benchmark_mode = False
        return self


@dataclass
class DerivedPPOLearnerConfig(Generic[ObsSpaceType, ActionSpaceType]):
    learner_config: PPOLearnerConfigModel
    agent_controller_name: str | None
    obs_space: ObsSpaceType
    action_space: ActionSpaceType
    checkpoint_load_folder: str | None = None


@dataclass
class PPOData:
    batch_consumption_time: float
    cumulative_model_updates: int
    actor_entropy: float
    kl_divergence: float
    critic_loss: float
    sb3_clip_fraction: float
    actor_update_magnitude: float
    critic_update_magnitude: float


ACTOR_CRITIC_FILE = "actor_critic.pt"
OPTIMIZERS_FILE = "optimizers.pt"
MISC_STATE = "misc.json"


class PPOLearner(
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        TrajectoryProcessorData,
    ]
):
    def __init__(
        self,
        actor_critic_factory: Callable[
            [ObsSpaceType, ActionSpaceType, torch.dtype, torch.device, str | None],
            ActorCritic[AgentID, ObsType, ActionType],
        ],
        optimizers_factory: Callable[
            [
                ActorCritic[AgentID, ObsType, ActionType],
                dict[str, dict[str, JsonValue]],
                str | None,
            ],
            list[torch.optim.Optimizer],
        ],
    ):
        self.actor_critic_factory: Callable[
            [ObsSpaceType, ActionSpaceType, torch.dtype, torch.device, str | None],
            ActorCritic[AgentID, ObsType, ActionType],
        ] = actor_critic_factory
        self.optimizers_factory: Callable[
            [
                ActorCritic[AgentID, ObsType, ActionType],
                dict[str, dict[str, JsonValue]],
                str | None,
            ],
            list[torch.optim.Optimizer],
        ] = optimizers_factory
        self.critic_loss_fn: nn.Module = torch.nn.MSELoss()
        self.config: DerivedPPOLearnerConfig[ObsSpaceType, ActionSpaceType]
        self.actor_critic: ActorCritic[AgentID, ObsType, ActionType]
        self.optimizers: list[torch.optim.Optimizer]
        self.cumulative_model_updates: int
        self.minibatch_size: int
        self.batch_advantages: torch.Tensor
        self.batch_old_probs: torch.Tensor
        self.batch_target_values: torch.Tensor

        self._log_prefix: str
        self._non_blocking: bool

    def load(self, config: DerivedPPOLearnerConfig[ObsSpaceType, ActionSpaceType]):
        self.config = config

        if (
            config.learner_config.cudnn_benchmark_mode
            and config.learner_config.device.type == "cuda"
        ):
            torch.backends.cudnn.benchmark = True

        self.actor_critic = self.actor_critic_factory(
            config.obs_space,
            config.action_space,
            config.learner_config.dtype,
            config.learner_config.device,
            config.agent_controller_name,
        )
        self.optimizers = self.optimizers_factory(
            self.actor_critic,
            self.config.learner_config.optimizer_named_parameter_group_kwargs,
            config.agent_controller_name,
        )
        self._log_prefix = (
            f"{config.agent_controller_name}:"
            if config.agent_controller_name is not None
            else ""
        )
        self._non_blocking = self.config.learner_config.device.type != "cpu"

        self.cumulative_model_updates = 0

        if self.config.checkpoint_load_folder is not None:
            # Save kwargs from optimizer factory
            optimizer_groups_kwargs = [
                [
                    {
                        k: v
                        for k, v in group.items()
                        if k not in ("params", "param_names")
                    }
                    for group in optimizer.param_groups
                ]
                for optimizer in self.optimizers
            ]
            self._load_from_checkpoint()
            # Put kwargs back into optimizers after they were overwritten by checkpoint
            for optimizer, groups_kwargs in zip(
                self.optimizers, optimizer_groups_kwargs
            ):
                for group, group_kwargs in zip(optimizer.param_groups, groups_kwargs):
                    group.update(group_kwargs)

        self.minibatch_size = int(
            np.ceil(
                self.config.learner_config.batch_size
                / self.config.learner_config.n_minibatches
            )
        )
        self.batch_advantages = torch.empty(
            self.config.learner_config.batch_size,
            dtype=config.learner_config.dtype,
            device=config.learner_config.device,
        )
        self.batch_old_probs = torch.empty(
            self.config.learner_config.batch_size,
            dtype=config.learner_config.dtype,
            device=config.learner_config.device,
        )
        self.batch_target_values = torch.empty(
            self.config.learner_config.batch_size,
            dtype=config.learner_config.dtype,
            device=config.learner_config.device,
        )

    def _load_from_checkpoint(self):
        assert self.config.checkpoint_load_folder is not None, (
            "Cannot load from checkpoint if checkpoint load folder is None!"
        )

        assert os.path.exists(self.config.checkpoint_load_folder), (
            f"{self._log_prefix} PPO Learner cannot find folder: {self.config.checkpoint_load_folder}"
        )

        _ = self.actor_critic.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, ACTOR_CRITIC_FILE),
                map_location=self.config.learner_config.device,
            )
        )
        optimizer_state_dicts = torch.load(
            os.path.join(self.config.checkpoint_load_folder, OPTIMIZERS_FILE),
            map_location=self.config.learner_config.device,
        )
        for optimizer, state_dict in zip(self.optimizers, optimizer_state_dicts):
            optimizer.load_state_dict(state_dict)
        try:
            with open(
                os.path.join(self.config.checkpoint_load_folder, MISC_STATE), "rt"
            ) as f:
                misc_state = json.load(f)
                self.cumulative_model_updates = misc_state["cumulative_model_updates"]
        except FileNotFoundError:
            print(
                f"{self._log_prefix} Tried to load the PPO learner's misc state from the file at location {os.path.join(self.config.checkpoint_load_folder, MISC_STATE)}, but there is no such file! Miscellaneous stats will be initialized as if this were a new run instead."
            )
            self.cumulative_model_updates = 0

    def save_checkpoint(self, folder_path: str | os.PathLike[str]) -> None:
        os.makedirs(folder_path, exist_ok=True)
        torch.save(
            self.actor_critic.state_dict(), os.path.join(folder_path, ACTOR_CRITIC_FILE)
        )
        torch.save(
            [optimizer.state_dict() for optimizer in self.optimizers],
            os.path.join(folder_path, OPTIMIZERS_FILE),
        )
        with open(os.path.join(folder_path, MISC_STATE), "wt") as f:
            json.dump(
                {"cumulative_model_updates": self.cumulative_model_updates}, f, indent=4
            )

    def learn(
        self,
        exp: ExperienceBuffer[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            ObsSpaceType,
            ActionSpaceType,
            TrajectoryProcessorData,
        ],
    ):
        """
        Compute PPO updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.
            collect_metrics_fn: Function to be called with the PPO metrics resulting from learn()
        """

        n_batches = 0
        clip_fractions: list[tuple[torch.Tensor, float]] = []
        entropies: list[torch.Tensor] = []
        divergences: list[torch.Tensor] = []
        val_losses: list[torch.Tensor] = []

        # Save parameters before computing any updates.
        actor_before = self.actor_critic.get_actor_parameter_vector()
        critic_before = self.actor_critic.get_critic_parameter_vector()

        t1 = time.time()
        for _epoch in range(self.config.learner_config.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(
                self.config.learner_config.batch_size
            )
            for batch in batches:
                (
                    batch_agent_ids,
                    batch_obs,
                    batch_acts,
                    _batch_old_probs,
                    _batch_values,
                    _batch_advantages,
                ) = batch

                _ = self.batch_old_probs.copy_(
                    _batch_old_probs, non_blocking=self._non_blocking
                )
                _ = self.batch_target_values.copy_(
                    _batch_values, non_blocking=self._non_blocking
                )
                _ = self.batch_advantages.copy_(
                    _batch_advantages, non_blocking=self._non_blocking
                )
                _ = self.batch_target_values.add_(self.batch_advantages)

                if self.config.learner_config.advantage_standardization:
                    std, mean = torch.std_mean(self.batch_advantages)
                    if torch.isnan(std):
                        std = torch.tensor(1, dtype=self.config.learner_config.dtype)
                    _ = self.batch_advantages.sub_(mean).div_(std + 1e-8)

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                for minibatch_slice in range(
                    0, self.config.learner_config.batch_size, self.minibatch_size
                ):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = min(
                        start + self.minibatch_size,
                        self.config.learner_config.batch_size,
                    )
                    n = stop - start
                    minibatch_ratio = n / self.config.learner_config.batch_size

                    agent_ids = batch_agent_ids[start:stop]
                    obs = batch_obs[start:stop]
                    acts = batch_acts[start:stop]
                    old_probs = self.batch_old_probs[start:stop]
                    target_values = self.batch_target_values[start:stop]
                    advantages = self.batch_advantages[start:stop]

                    log_probs, entropy, vals = self.actor_critic.get_backprop_data(
                        agent_ids, obs, acts
                    )
                    log_probs = log_probs.view_as(old_probs)
                    entropy = entropy * minibatch_ratio
                    vals = vals.view_as(target_values)

                    # Compute PPO loss.
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.learner_config.clip_range,
                        1.0 + self.config.learner_config.clip_range,
                    )

                    # Compute KL divergence & clip fraction using SB3 method for reporting.
                    with torch.no_grad():
                        log_ratio = log_probs - old_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach() * minibatch_ratio

                        # From the stable-baselines3 implementation of PPO.
                        clip_fraction = torch.mean(
                            (
                                torch.abs(ratio - 1)
                                > self.config.learner_config.clip_range
                            ).float()
                        ).to(device="cpu", non_blocking=self._non_blocking)
                        clip_fractions.append((clip_fraction, minibatch_ratio))

                    actor_loss = (
                        -torch.min(ratio * advantages, clipped * advantages).mean()
                        * minibatch_ratio
                    )
                    value_loss: torch.Tensor = (
                        self.critic_loss_fn(vals, target_values) * minibatch_ratio
                    )
                    ppo_loss = (
                        actor_loss - entropy * self.config.learner_config.ent_coef
                    )

                    total_loss = ppo_loss + value_loss
                    total_loss.backward()  # pyright: ignore [reportUnknownMemberType, reportUnusedCallResult]

                    val_losses.append(
                        value_loss.to(
                            device="cpu", non_blocking=self._non_blocking
                        ).detach()
                    )
                    divergences.append(
                        kl.to(device="cpu", non_blocking=self._non_blocking).detach()
                    )
                    entropies.append(
                        entropy.to(
                            device="cpu", non_blocking=self._non_blocking
                        ).detach()
                    )

                if self.config.learner_config.max_grad_norm is not None:
                    _ = torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        max_norm=self.config.learner_config.max_grad_norm,
                    )

                for optimizer in self.optimizers:
                    optimizer.step()

                n_batches += 1

        # Compute magnitude of updates made to the actor and critic.
        actor_after = self.actor_critic.get_actor_parameter_vector()
        critic_after = self.actor_critic.get_critic_parameter_vector()
        actor_update_magnitude = cast(
            float,
            (actor_before - actor_after).norm().cpu().item(),  # pyright: ignore [reportUnknownMemberType]
        )
        critic_update_magnitude = cast(
            float,
            (critic_before - critic_after).norm().cpu().item(),  # pyright: ignore [reportUnknownMemberType]
        )

        # synchronize to finalize the values sent to cpu without blocking for PPOData
        if self.config.learner_config.device.type == "cuda":
            torch.cuda.synchronize(device=self.config.learner_config.device)

        tot_clip = sum(
            v.item() * minibatch_ratio for (v, minibatch_ratio) in clip_fractions
        )
        tot_entropy = sum(v.item() for v in entropies)
        tot_divergence = sum(v.item() for v in divergences)
        tot_val_loss = sum(v.item() for v in val_losses)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        self.cumulative_model_updates += n_batches

        # If there were no batches, we just want to log the total time spent here (and totals will all be 0 anyway), so just set n_batches to 1
        if n_batches == 0:
            n_batches = 1
        mean_clip = tot_clip / n_batches
        mean_entropy = tot_entropy / n_batches
        mean_divergence = tot_divergence / n_batches
        mean_val_loss = tot_val_loss / n_batches
        return PPOData(
            (time.time() - t1) / n_batches,
            self.cumulative_model_updates,
            mean_entropy,
            mean_divergence,
            mean_val_loss,
            mean_clip,
            actor_update_magnitude,
            critic_update_magnitude,
        )
