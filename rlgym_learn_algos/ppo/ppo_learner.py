import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Optional

import numpy as np
import torch
from pydantic import BaseModel, field_serializer, model_validator
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import nn as nn

from rlgym_learn_algos.util.torch_functions import get_device
from rlgym_learn_algos.util.torch_pydantic import (
    PydanticTorchDevice,
    PydanticTorchDtype,
)

from .actor import Actor
from .critic import Critic
from .experience_buffer import ExperienceBuffer
from .trajectory_processor import TrajectoryProcessorConfig, TrajectoryProcessorData


class PPOLearnerConfigModel(BaseModel, extra="forbid"):
    dtype: PydanticTorchDtype = torch.float32
    n_epochs: int = 1
    batch_size: int = 50000
    n_minibatches: int = 1
    ent_coef: float = 0.005
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    advantage_normalization: bool = True
    device: PydanticTorchDevice = "auto"
    cudnn_benchmark_mode: bool = True

    @model_validator(mode="before")
    @classmethod
    def set_device(cls, data):
        if isinstance(data, dict):
            if "device" not in data:
                data["device"] = "auto"
            data["device"] = get_device(data["device"])
        return data

    @model_validator(mode="after")
    def validate_cudnn_benchmark(self):
        if self.device.type != "cuda":
            self.cudnn_benchmark_mode = False
        return self


@dataclass
class DerivedPPOLearnerConfig:
    learner_config: PPOLearnerConfigModel
    agent_controller_name: str
    obs_space: ObsSpaceType
    action_space: ActionSpaceType
    checkpoint_load_folder: Optional[str] = None


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


ACTOR_FILE = "actor.pt"
ACTOR_OPTIMIZER_FILE = "actor_optimizer.pt"
CRITIC_FILE = "critic.pt"
CRITIC_OPTIMIZER_FILE = "critic_optimizer.pt"
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
        actor_factory: Callable[
            [ObsSpaceType, ActionSpaceType, torch.device],
            Actor[AgentID, ObsType, ActionType],
        ],
        critic_factory: Callable[
            [ObsSpaceType, torch.device], Critic[AgentID, ObsType]
        ],
    ):
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory

    def load(self, config: DerivedPPOLearnerConfig):
        self.config = config

        if (
            config.learner_config.cudnn_benchmark_mode
            and config.learner_config.device.type == "cuda"
        ):
            torch.backends.cudnn.benchmark = True

        self.actor = self.actor_factory(
            config.obs_space, config.action_space, config.learner_config.device
        )
        self.critic = self.critic_factory(
            config.obs_space, config.learner_config.device
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learner_config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learner_config.critic_lr
        )
        self.critic_loss_fn = torch.nn.MSELoss()

        # Calculate parameter counts
        actor_params = self.actor.parameters()
        critic_params = self.critic.parameters()

        trainable_actor_parameters = filter(lambda p: p.requires_grad, actor_params)
        actor_params_count = sum(p.numel() for p in trainable_actor_parameters)

        trainable_critic_parameters = filter(lambda p: p.requires_grad, critic_params)
        critic_params_count = sum(p.numel() for p in trainable_critic_parameters)

        total_parameters = actor_params_count + critic_params_count

        # Display in a structured manner
        print(f"{self.config.agent_controller_name}: Trainable Parameters:")
        print(f"{self.config.agent_controller_name}: {'Component':<10} {'Count':<10}")
        print("-" * 20)
        print(
            f"{self.config.agent_controller_name}: {'Policy':<10} {actor_params_count:<10}"
        )
        print(
            f"{self.config.agent_controller_name}: {'Critic':<10} {critic_params_count:<10}"
        )
        print("-" * 20)
        print(
            f"{self.config.agent_controller_name}: {'Total':<10} {total_parameters:<10}"
        )

        print(
            f"{self.config.agent_controller_name}: Current Policy Learning Rate: {self.config.learner_config.actor_lr}"
        )
        print(
            f"{self.config.agent_controller_name}: Current Critic Learning Rate: {self.config.learner_config.critic_lr}"
        )
        self.cumulative_model_updates = 0

        if self.config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        self.minibatch_size = int(
            np.ceil(
                self.config.learner_config.batch_size
                / self.config.learner_config.n_minibatches
            )
        )

    def _load_from_checkpoint(self):

        assert os.path.exists(
            self.config.checkpoint_load_folder
        ), f"{self.config.agent_controller_name}: PPO Learner cannot find folder: {self.config.checkpoint_load_folder}"

        self.actor.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, ACTOR_FILE),
                map_location=self.config.learner_config.device,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, CRITIC_FILE),
                map_location=self.config.learner_config.device,
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, ACTOR_OPTIMIZER_FILE),
                map_location=self.config.learner_config.device,
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, CRITIC_OPTIMIZER_FILE),
                map_location=self.config.learner_config.device,
            )
        )
        try:
            with open(
                os.path.join(self.config.checkpoint_load_folder, MISC_STATE), "rt"
            ) as f:
                misc_state = json.load(f)
                self.cumulative_model_updates = misc_state["cumulative_model_updates"]
        except FileNotFoundError:
            print(
                f"{self.config.agent_controller_name}: Tried to load the PPO learner's misc state from the file at location {str(os.path.join(self.config.checkpoint_load_folder, MISC_STATE))}, but there is no such file! Miscellaneous stats will be initialized as if this were a new run instead."
            )
            self.cumulative_model_updates = 0

    def save_checkpoint(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder_path, ACTOR_FILE))
        torch.save(self.critic.state_dict(), os.path.join(folder_path, CRITIC_FILE))
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(folder_path, ACTOR_OPTIMIZER_FILE),
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(folder_path, CRITIC_OPTIMIZER_FILE),
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
        clip_fractions = []
        entropies = []
        divergences = []
        val_losses = []

        # Save parameters before computing any updates.
        actor_before = torch.nn.utils.parameters_to_vector(self.actor.parameters())
        critic_before = torch.nn.utils.parameters_to_vector(self.critic.parameters())

        t1 = time.time()
        for epoch in range(self.config.learner_config.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(
                self.config.learner_config.batch_size
            )
            for batch in batches:
                (
                    batch_agent_ids,
                    batch_obs,
                    batch_acts,
                    batch_old_probs,
                    batch_values,
                    batch_advantages,
                ) = batch
                batch_target_values = batch_values + batch_advantages
                if self.config.learner_config.advantage_normalization:
                    old_device = batch_advantages.device
                    batch_advantages = batch_advantages.to(
                        self.config.learner_config.device
                    )
                    std, mean = torch.std_mean(batch_advantages)
                    batch_advantages = (batch_advantages - mean) / (std + 1e-8)
                    batch_advantages = batch_advantages.to(old_device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                for minibatch_slice in range(
                    0, self.config.learner_config.batch_size, self.minibatch_size
                ):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = min(
                        start + self.minibatch_size,
                        self.config.learner_config.batch_size,
                    )
                    minibatch_ratio = (
                        stop - start
                    ) / self.config.learner_config.batch_size

                    agent_ids = batch_agent_ids[start:stop]
                    obs = batch_obs[start:stop]
                    acts = batch_acts[start:stop]
                    advantages = batch_advantages[start:stop].to(
                        self.config.learner_config.device
                    )
                    old_probs = batch_old_probs[start:stop].to(
                        self.config.learner_config.device
                    )
                    target_values = batch_target_values[start:stop].to(
                        self.config.learner_config.device
                    )

                    # Compute value estimates.
                    vals = self.critic(agent_ids, obs).view_as(target_values)

                    # Get actor log probs & entropy.
                    log_probs, entropy = self.actor.get_backprop_data(
                        agent_ids, obs, acts
                    )
                    log_probs = log_probs.view_as(old_probs)
                    entropy = entropy * minibatch_ratio

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
                        ).to(device="cpu", non_blocking=True)
                        clip_fractions.append((clip_fraction, minibatch_ratio))

                    actor_loss = (
                        -torch.min(ratio * advantages, clipped * advantages).mean()
                        * minibatch_ratio
                    )
                    value_loss = (
                        self.critic_loss_fn(vals, target_values) * minibatch_ratio
                    )
                    ppo_loss = (
                        actor_loss - entropy * self.config.learner_config.ent_coef
                    )

                    ppo_loss.backward()
                    value_loss.backward()

                    val_losses.append(
                        value_loss.to(device="cpu", non_blocking=True).detach()
                    )
                    divergences.append(kl.to(device="cpu", non_blocking=True).detach())
                    entropies.append(
                        entropy.to(device="cpu", non_blocking=True).detach()
                    )

                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                n_batches += 1

        # Compute magnitude of updates made to the actor and critic.
        actor_after = torch.nn.utils.parameters_to_vector(self.actor.parameters())
        critic_after = torch.nn.utils.parameters_to_vector(self.critic.parameters())
        actor_update_magnitude = (actor_before - actor_after).norm().cpu().item()
        critic_update_magnitude = (critic_before - critic_after).norm().cpu().item()

        if self.config.learner_config.device.type != "cpu":
            torch.cuda.current_stream().synchronize()

        tot_clip = sum(
            v.item() * minibatch_ratio for (v, minibatch_ratio) in clip_fractions
        )
        tot_entropy = sum(v.item() for v in entropies)
        tot_divergence = sum(v.item() for v in divergences)
        tot_val_loss = sum(v.item() for v in val_losses)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

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
