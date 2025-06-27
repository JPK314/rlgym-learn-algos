"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rlgym.api import AgentID
from tensordict import NonTensorStack, TensorDict
from torch.distributions.utils import probs_to_logits

from .actor import Actor


class DiscreteFF(Actor[AgentID, np.ndarray, np.ndarray]):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_sizes: Iterable[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_size, layer_sizes[0], dtype=dtype), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size, dtype=dtype))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

    def get_action(
        self, agent_id_list, obs_list, **kwargs
    ) -> Tuple[Iterable[np.ndarray], TensorDict]:
        obs = torch.as_tensor(np.array(obs_list), dtype=self.dtype, device=self.device)
        probs = self.model(obs)
        probs = torch.clamp(probs, min=1e-11, max=1)
        if "deterministic" in kwargs and kwargs["deterministic"]:
            action = probs.cpu().numpy().argmax(axis=-1)
            return action, torch.zeros(len(agent_id_list))

        action = torch.multinomial(probs, 1, True)
        log_prob: torch.Tensor = torch.log(probs).gather(-1, action)

        aald = TensorDict(
            agent_ids=NonTensorStack(*agent_id_list),
            obs=obs,
            action=action,
            log_prob=log_prob,
            device=self.device,
            batch_size=[len(agent_id_list)],
        )

        return action.cpu().numpy(), aald

    def get_backprop_data(self, agent_id_list, obs, action, **kwargs):
        probs = self.model(obs.to(self.device))
        probs = torch.clamp(probs, min=1e-11, max=1)
        logits = probs_to_logits(probs)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        entropy = -(logits * probs).sum(dim=-1)
        action_logits = logits.gather(-1, action.to(self.device))

        return action_logits, entropy.mean()
