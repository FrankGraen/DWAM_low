from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def is_success(
    env,
    command_name: str = "flag_trajectory_finished",
) -> torch.Tensor:
    """
    Determine if the box has reached the target position and orientation with low speed.
    """
    # Get the box's relative target command (goal_pose_b: [dx, dy, dz, qw, qx, qy, qz] in box body coordinates)
    if not hasattr(env, command_name):
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        finished = getattr(env, command_name)
    return finished
