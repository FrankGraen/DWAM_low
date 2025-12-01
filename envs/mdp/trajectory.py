from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .trajectory_cfg import TRAJECTORY_TYPE
from .trajectory_cfg import TrajectoryGenerator
from .trajectory_cfg import TrajectoryConfig


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def init_trajectory(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Initialize trajectory generator for the environment."""
    cfg = TrajectoryConfig()
    env.trajectory_generator = TrajectoryGenerator(cfg)
    box_pos = env.scene["box_1"].data.body_link_state_w[:, 0, :3]
    new_trajectory = env.trajectory_generator.generate_trajectories(box_pos)
    env.current_trajectories = new_trajectory