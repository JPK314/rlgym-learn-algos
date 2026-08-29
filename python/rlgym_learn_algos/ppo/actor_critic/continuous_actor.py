"""
File: continuous_policy.py
Author: Jonathan Keegan

Description:
    An implementation of a policy for the continuous action space. The network is implemented as a dense feed-forward
    network with 2N outputs - N means and N standard deviations - which are used to sample N independent Gaussians.
    To keep the variance positive, a softplus is applied to the last linear layer for the N standard deviation
    outputs. The sampled output is then squashed into a given range in each dimension using tanh and an affine map.

"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import torch
from rlgym.api import AgentID
from torch import nn
from torch.distributions import Normal
from typing_extensions import override

from .actor import Actor

MODE_CONVERGENCE_ITERATIONS = 32


# AI DISCLOSURE: The below method was written by ChatGPT Sol 5.6 Medium.
def _tanh_normal_mode_kernel(
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Find a mode of Y = tanh(U), where independent scalar variables satisfy

        U_i ~ Normal(mean_i, std_i**2).

    `mean` and `std` may have any broadcast-compatible shapes. The returned
    tensor has their broadcasted shape.

    If the transformed distribution is symmetric and bimodal (mean == 0),
    the positive mode is returned.
    """
    mean, std = cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.broadcast_tensors(mean, std),  # pyright: ignore [reportUnknownMemberType]
    )

    variance = std.square()

    # By symmetry, solve the problem for abs(mean), selecting the positive
    # mode, and then restore the sign. For mean == 0, choose the positive mode.
    positive_mean = mean.abs()

    # Stationary points satisfy
    #
    #     g(u) = u - mean - 2 variance tanh(u) = 0.
    #
    # When variance > 1/2, g is guaranteed to be increasing to the right of
    #
    #     turning_point = acosh(sqrt(2 variance)).
    #
    # The positive/global mode lies to the right of both `mean` and this
    # turning point.
    turning_point = torch.acosh(torch.sqrt(torch.clamp(2.0 * variance, min=1.0)))

    lower = torch.maximum(positive_mean, turning_point)
    upper = positive_mean + 2.0 * variance

    # On [lower, upper], g is monotonic increasing, with
    #
    #     g(lower) <= 0
    #     g(upper) >= 0.
    #
    # A fixed iteration count avoids GPU-to-CPU synchronization from
    # convergence checks and works well with torch.compile.
    for _ in range(MODE_CONVERGENCE_ITERATIONS):
        midpoint = (lower + upper) * 0.5
        value = midpoint - positive_mean - 2.0 * variance * torch.tanh(midpoint)

        lower = torch.where(value <= 0, midpoint, lower)
        upper = torch.where(value > 0, midpoint, upper)

    positive_pre_tanh_mode = (lower + upper) * 0.5

    sign = torch.where(
        mean < 0,
        -torch.ones_like(mean),
        torch.ones_like(mean),
    )
    pre_tanh_mode = sign * positive_pre_tanh_mode

    return torch.tanh(pre_tanh_mode)


class ContinuousActor(Actor[AgentID, np.ndarray, np.ndarray]):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        output_ranges: tuple[tuple[float, float], ...],
        layer_sizes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        assert output_shape == len(output_ranges), (
            "One range must be provided for each output dimension."
        )
        assert all(low < high for (low, high) in output_ranges), (
            "Ranges must be tuples of the form (low, high) where low < high."
        )
        super().__init__()
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        # Build the neural network.
        assert len(layer_sizes) != 0, (
            "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        )
        layers: list[nn.Module] = [
            nn.Linear(input_shape, layer_sizes[0], dtype=dtype),
            nn.ReLU(),
        ]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size, dtype=dtype))
            layers.append(nn.ReLU())
            prev_size = size

        self.trunk: nn.Module = nn.Sequential(*layers).to(device)
        self.mean_head: nn.Module = nn.Linear(
            layer_sizes[-1], output_shape, dtype=dtype
        ).to(device)
        self.std_head: nn.Module = nn.Sequential(
            nn.Linear(layer_sizes[-1], output_shape, dtype=dtype), nn.Softplus()
        ).to(device)
        self.affine_m: torch.Tensor = torch.tensor(
            [(high - low) / 2 for (low, high) in output_ranges],
            dtype=dtype,
            device=device,
        )
        self.affine_b: torch.Tensor = torch.tensor(
            [(high + low) / 2 for (low, high) in output_ranges],
            dtype=dtype,
            device=device,
        )
        self._inv_affine_m: torch.Tensor = torch.div(1, self.affine_m)
        self._log_pdf_const: torch.Tensor = torch.log(self.affine_m) + 0.5 * math.log(
            2 * math.pi
        )
        self._inv_tanh_clamp: tuple[float, float] = (
            -1.0 + torch.finfo(self.dtype).eps,
            1.0 - torch.finfo(self.dtype).eps,
        )

    def logpdf(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        gaussian_sample: torch.Tensor | None = None,
        tanh_sample: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ):
        """
        Function to compute the log of the pdf of our distribution parameterized by (mean, std) evaluated at action.
        :param mean: Mean of the distribution to evaluate.
        :param std: Diagonal of the covariance matrix of the distribution to evaluate.
        :param gaussian_sample: Pre-tanh sample of gaussian parameterized by mean, std (will be computed if not passed)
        :param tanh_sample: Pre-affine-transformation sample of tanh'd gaussian parameterized by mean, std (will be computed if not passed)
        :param action: Value to compute the logpdf for (will be computed using tanh_sample or gaussian_sample if not passed)
        :return: ln(pdf(x)).
        """
        if tanh_sample is None:
            if action is not None:
                tanh_sample = (action - self.affine_b) * self._inv_affine_m
            elif gaussian_sample is not None:
                tanh_sample = torch.tanh(gaussian_sample)
            else:
                raise ValueError(
                    "If tanh_sample is not passed, action or gaussian_sample must be passed as non-None"
                )

        if gaussian_sample is None:
            x = tanh_sample.clamp(*self._inv_tanh_clamp)
            # Taken from Pyro: https://github.com/pyro-ppl/pyro
            gaussian_sample = 0.5 * (x.log1p() - (-x).log1p())

        term1 = torch.divide((gaussian_sample - mean).square(), (2 * std.square()))
        term2 = torch.log(std)
        term3 = self._log_pdf_const
        term4 = torch.log(1 - tanh_sample.square())

        result = -(term1 + term2 + term3 + term4)

        if len(result.shape) > 1:
            log_prob = result.sum(dim=-1)
        else:
            log_prob = result.sum()
        return log_prob

    def get_output(
        self, obs_list: Sequence[np.ndarray] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs_list, torch.Tensor):
            obs = obs_list
        else:
            obs = torch.as_tensor(
                np.asarray(obs_list), dtype=self.dtype, device=self.device
            )
        trunk_output = self.trunk(obs)
        return self.mean_head(trunk_output), self.std_head(trunk_output)

    @override
    def get_actions(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[np.ndarray] | torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[Iterable[np.ndarray], torch.Tensor]:
        # This should work for both inputs of the form (B, input_shape) and inputs of the form (input_shape,)
        with torch.no_grad():
            mean, std = self.get_output(obs_list)
        if kwargs.get("deterministic"):
            tanh_sample = _tanh_normal_mode_kernel(mean, std)
            action = tanh_sample * self.affine_m + self.affine_b
            # The probability of a deterministic action occurring is 1 -> log(1) = 0.

            return action.cpu().numpy(), torch.zeros(
                len(obs_list), dtype=self.dtype, device="cpu"
            )

        distribution = Normal(loc=mean, scale=std)
        gaussian_sample = distribution.sample()
        tanh_sample = torch.tanh(gaussian_sample)
        action = tanh_sample * self.affine_m + self.affine_b

        return action.cpu().numpy(), self.logpdf(
            mean, std, gaussian_sample, tanh_sample
        ).cpu().squeeze()

    @override
    def get_backprop_data(
        self,
        agent_id_list: Sequence[AgentID],
        obs_list: Sequence[np.ndarray] | torch.Tensor,
        action_list: Sequence[np.ndarray] | torch.Tensor,
        **kwargs: dict[str, Any],
    ):
        mean, std = self.get_output(obs_list)

        if isinstance(action_list, torch.Tensor):
            actions_tensor = action_list
        else:
            actions_tensor = torch.as_tensor(
                np.asarray(action_list), dtype=self.dtype, device=self.device
            )

        log_probs = self.logpdf(mean, std, action=actions_tensor)
        # No analytical form for entropy, just approximate by sampling from distribution. Not using actions_tensor because that may not be in distribution anymore
        distribution = Normal(loc=mean, scale=std)
        gaussian_sample = distribution.sample()
        entropy = -self.logpdf(mean, std, gaussian_sample=gaussian_sample).mean()

        return log_probs, entropy
