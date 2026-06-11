"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rlgym.api import AgentID
from torch.distributions.utils import probs_to_logits
from typing_extensions import override

from .actor import Actor


class DiscreteFF(Actor[AgentID, np.ndarray, np.ndarray]):
    def __init__(
        self,
        input_size: int,
        n_actions: int,
        layer_sizes: tuple[int],
        device: torch.Device,
    ):
        super().__init__()
        self.device: torch.Device = device

        assert len(layer_sizes) != 0, (
            "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        )
        layers: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model: nn.Module = nn.Sequential(*layers).to(self.device)

        self.n_actions: int = n_actions

    def get_output(self, obs_list: Sequence[np.ndarray]) -> torch.Tensor:
        obs = torch.as_tensor(
            np.array(obs_list), dtype=torch.float32, device=self.device
        )
        probs = self.model(obs)
        probs = torch.clamp(probs, min=1e-11, max=1)
        return probs

    @override
    def get_action(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[np.ndarray],
        **kwargs: dict[str, Any],
    ) -> tuple[Iterable[np.ndarray], torch.Tensor]:
        probs = self.get_output(obs_list)
        if "deterministic" in kwargs and kwargs["deterministic"]:
            action = probs.cpu().numpy().argmax(axis=-1)
            return action, torch.zeros(action.shape)

        action = torch.multinomial(probs, 1, True)
        log_prob: torch.Tensor = torch.log(probs).gather(-1, action)

        return action.cpu().numpy(), log_prob.squeeze().to(
            device="cpu", non_blocking=True
        )

    @override
    def get_backprop_data(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[np.ndarray],
        acts: Sequence[np.ndarray],
        **kwargs: dict[str, Any],
    ):
        probs = self.get_output(obs_list)
        acts_tensor = torch.as_tensor(np.array(acts)).to(self.device)
        logits = probs_to_logits(probs)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        entropy = -(logits * probs).sum(dim=-1)
        action_logits = logits.gather(-1, acts_tensor)

        return action_logits.to(self.device), entropy.to(self.device).mean()
