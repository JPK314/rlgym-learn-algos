from typing import Generic

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from rlgym_learn import Timestep

from .trajectory import Trajectory


class EnvTrajectories(Generic[AgentID, ObsType, ActionType, RewardType]):
    def __init__(
        self,
        env_agent_ids: list[AgentID],
    ) -> None:
        self.used_agent_id_idx_map: dict[AgentID, int] = dict(
            zip(env_agent_ids, range(len(env_agent_ids)))
        )
        self.obs_lists: dict[AgentID, list[ObsType]] = {}
        self.action_lists: dict[AgentID, list[ActionType]] = {}
        self.reward_lists: dict[AgentID, list[RewardType]] = {}
        self.final_obs: dict[AgentID, ObsType | None] = {}
        self.dones: dict[AgentID, bool] = {}
        self.truncateds: dict[AgentID, bool] = {}
        for agent_id in self.used_agent_id_idx_map:
            self.obs_lists[agent_id] = []
            self.action_lists[agent_id] = []
            self.reward_lists[agent_id] = []
            self.final_obs[agent_id] = None
            self.dones[agent_id] = False
            self.truncateds[agent_id] = False
        self.log_probs_list: list[list[np.ndarray]] = []

    def add_steps(
        self,
        controlled_agents: list[AgentID],
        timesteps: list[Timestep[AgentID, ObsType, ActionType, RewardType]],
        log_probs: list[np.ndarray],
    ):
        # Intersect used_agent_id_idx_map's keys with controlled_agents. We can't learn from agents we only partially controlled.
        steps_removed = 0
        for agent_id in self.used_agent_id_idx_map:
            if agent_id not in controlled_agents:
                del self.used_agent_id_idx_map[agent_id]
                obs_list = self.obs_lists.pop(agent_id)
                steps_removed += len(obs_list)
                del self.action_lists[agent_id]
                del self.reward_lists[agent_id]
                del self.final_obs[agent_id]
                del self.dones[agent_id]
                del self.truncateds[agent_id]

        steps_added = 0
        for timestep in timesteps:
            agent_id = timestep.agent_id
            # We only want to process the timesteps of agent ids we included from this env when creating the EnvTrajectories instance
            if agent_id not in self.used_agent_id_idx_map:
                continue
            if not self.dones[agent_id]:
                steps_added += 1
                self.obs_lists[agent_id].append(timestep.obs)
                self.action_lists[agent_id].append(timestep.action)
                self.reward_lists[agent_id].append(timestep.reward)
                self.final_obs[agent_id] = timestep.next_obs
                now_done = timestep.terminated or timestep.truncated
                if now_done:
                    self.dones[agent_id] = True
                    self.truncateds[agent_id] = timestep.truncated

        # We append all the log probs but we will deal with this later when getting trajectories
        self.log_probs_list.append(log_probs)
        return steps_added - steps_removed

    def finalize(self):
        """
        Truncates any unfinished trajectories, marks all trajectories as done.
        """
        for agent_id in self.used_agent_id_idx_map:
            self.truncateds[agent_id] = (
                self.truncateds[agent_id] or not self.dones[agent_id]
            )
            self.dones[agent_id] = True

    def get_trajectories(
        self,
    ) -> list[Trajectory[AgentID, ObsType, ActionType, RewardType]]:
        """
        :return: List of trajectories relevant to this env
        """
        log_probs = torch.tensor(np.array(self.log_probs_list))
        trajectories: list[Trajectory[AgentID, ObsType, ActionType, RewardType]] = []
        for agent_id, idx in self.used_agent_id_idx_map.items():
            obs_list = self.obs_lists[agent_id]
            trajectories.append(
                Trajectory(
                    agent_id,
                    obs_list,
                    self.action_lists[agent_id],
                    log_probs[: len(obs_list), idx],
                    self.reward_lists[agent_id],
                    None,
                    self.final_obs[agent_id],
                    torch.tensor(0, dtype=torch.float32),
                    self.truncateds[agent_id],
                )
            )
        return trajectories
