from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trajectory_cfg import TRAJECTORY_TYPE
from .trajectory_cfg import TrajectoryGenerator
from .trajectory_cfg import TrajectoryConfig


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def init_trajectory(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    box_name: str, 
) -> None:
    """Initialize trajectory generator for the environment."""
    cfg = TrajectoryConfig()
    env.trajectory_generator = TrajectoryGenerator(cfg)
    box_pos = env.scene[box_name].data.body_link_state_w[env_ids, 0, :3]
    new_trajectory = env.trajectory_generator.generate_trajectories(box_pos)    # 
    env.current_trajectories[env_ids] = new_trajectory