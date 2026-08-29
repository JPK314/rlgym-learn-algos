import numpy as np
from rlgym.api import AgentID
from typing_extensions import override

from rlgym_learn_algos.util.running_stats import WelfordRunningStat

from .obs_standardizer import ObsStandardizer


class NumpyObsStandardizer(ObsStandardizer[AgentID, np.ndarray]):
    def __init__(
        self, steps_per_obs_stats_update: int, steps_until_fixed: float = np.inf
    ):
        self.obs_stats: WelfordRunningStat | None = None
        self.steps_per_obs_stats_update: int = steps_per_obs_stats_update
        self.obs_stats_start_index: int = 0
        self.steps_until_fixed: float = steps_until_fixed
        self.steps: int = 0

    @override
    def standardize(
        self, agent_id_list: list[AgentID], obs_list: list[np.ndarray]
    ) -> list[np.ndarray]:
        obs_arr = np.array(obs_list)
        if self.obs_stats is None:
            obs = obs_list[0]
            self.obs_stats = WelfordRunningStat(obs.shape)
        if self.steps < self.steps_until_fixed:
            stats_update_batch = obs_arr[
                self.obs_stats_start_index :: self.steps_per_obs_stats_update, :
            ]
            self.obs_stats_start_index = (
                self.steps_per_obs_stats_update
                - 1
                - (
                    (len(obs_list) - self.obs_stats_start_index - 1)
                    % self.steps_per_obs_stats_update
                )
            )
            for sample in stats_update_batch:
                self.obs_stats.update(sample)
            self.steps += 1
        obs_arr_standardized = (obs_arr - self.obs_stats.mean) / self.obs_stats.std
        return [obs for obs in obs_arr_standardized]
