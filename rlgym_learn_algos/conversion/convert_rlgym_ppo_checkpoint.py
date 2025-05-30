import json
import os
import time
from typing import Optional


def convert_rlgym_ppo_checkpoint(
    rlgym_ppo_checkpoint_folder: str, out_folder: Optional[str]
):

    if out_folder is None:
        out_folder = f"rlgym_ppo_converted_checkpoint_{time.time_ns()}"
    print(f"Saving converted checkpoint to folder {out_folder}")

    os.makedirs(out_folder, exist_ok=True)

    PPO_FILES = [
        ("PPO_POLICY_OPTIMIZER.pt", "actor_optimizer.pt"),
        ("PPO_POLICY.pt", "actor.pt"),
        ("PPO_VALUE_NET_OPTIMIZER.pt", "critic_optimizer.pt"),
        ("PPO_VALUE_NET.pt", "critic.pt"),
    ]
    os.makedirs(f"{out_folder}/ppo_learner", exist_ok=True)
    for file in PPO_FILES:
        with open(f"{rlgym_ppo_checkpoint_folder}/{file[0]}", "rb") as fin:
            with open(f"{out_folder}/ppo_learner/{file[1]}", "wb") as fout:
                fout.write(fin.read())
