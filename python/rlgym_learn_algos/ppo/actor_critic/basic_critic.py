from collections.abc import Sequence

import numpy as np
import torch
from rlgym.api import AgentID
from torch import nn
from typing_extensions import override

from .critic import Critic


class BasicCritic(Critic[AgentID, np.ndarray]):
    def __init__(
        self,
        input_size: int,
        layer_sizes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        assert len(layer_sizes) != 0, (
            "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        )
        layers = [nn.Linear(input_size, layer_sizes[0], dtype=dtype), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size, dtype=dtype))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1, dtype=dtype))
        self.model: nn.Module = nn.Sequential(*layers).to(self.device)

    @override
    def forward(
        self, agent_id_list: Sequence[AgentID], obs_list: Sequence[np.ndarray]
    ) -> torch.Tensor:
        obs = torch.as_tensor(np.array(obs_list), dtype=self.dtype, device=self.device)
        return self.model(obs)
