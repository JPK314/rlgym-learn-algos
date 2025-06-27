from typing import Callable, Dict, Generic, List, Optional

import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from rlgym_learn import Timestep
from tensordict import TensorDict

from .trajectory import Trajectory


class EnvTrajectories(Generic[AgentID, ObsType, ActionType, RewardType]):
    """
    Manage trajectories for all agents in an environment. Assumes agent id does not change during an episode.
    """

    def __init__(
        self,
        agent_ids: List[AgentID],
        agent_choice_fn: Callable[
            [List[AgentID]], List[int]
        ] = lambda agent_id_list: list(range(len(agent_id_list))),
    ) -> None:
        self.used_agent_id_idx_map = {
            agent_ids[idx]: idx for idx in agent_choice_fn(agent_ids)
        }
        self.reward_lists: Dict[AgentID, List[RewardType]] = {}
        self.final_obs: Dict[AgentID, Optional[ObsType]] = {}
        self.dones: Dict[AgentID, bool] = {}
        self.truncateds: Dict[AgentID, bool] = {}
        for agent_id in self.used_agent_id_idx_map:
            self.reward_lists[agent_id] = []
            self.final_obs[agent_id] = None
            self.dones[agent_id] = False
            self.truncateds[agent_id] = False
        self.aald_list = []

    def add_steps(self, timesteps: List[Timestep], aald: TensorDict):
        steps_added = 0
        for timestep in timesteps:
            agent_id = timestep.agent_id
            # We only want to process the timesteps of agent ids we included from this env when creating the EnvTrajectories instance
            if agent_id not in self.used_agent_id_idx_map:
                continue
            if not self.dones[agent_id]:
                steps_added += 1
                self.reward_lists[agent_id].append(timestep.reward)
                self.final_obs[agent_id] = timestep.next_obs
                now_done = timestep.terminated or timestep.truncated
                if now_done:
                    self.dones[agent_id] = True
                    self.truncateds[agent_id] = timestep.truncated
        # We append all the aald but we will deal with this later when getting trajectories
        self.aald_list.append(aald)
        return steps_added

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
    ) -> List[Trajectory[AgentID, ObsType, ActionType, RewardType]]:
        """
        :return: List of trajectories relevant to this env
        """
        aald = torch.stack(self.aald_list)
        trajectories = []
        for agent_id, idx in self.used_agent_id_idx_map.items():
            reward_list = self.reward_lists[agent_id]
            trajectories.append(
                Trajectory(
                    aald[: len(reward_list), idx],
                    self.reward_lists[agent_id],
                    None,
                    self.final_obs[agent_id],
                    torch.tensor(0, dtype=torch.float32),
                    self.truncateds[agent_id],
                )
            )
        return trajectories
