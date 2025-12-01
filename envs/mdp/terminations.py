from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def is_success(
    env,
    command_name: str = "box_follow_trajectory_finished",
    box_name: str = "box_1",
    progress_threshold: float = 0.99,
    distance_threshold: float = 0.2,
    speed_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Determine if the box has reached the target position and orientation with low speed.
    """
    # Get the box's relative target command (goal_pose_b: [dx, dy, dz, qw, qx, qy, qz] in box body coordinates)
    if not hasattr(env, 'trajectory_progress'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # 获取轨迹信息
    progress = env.trajectory_progress  # (num_envs,)
    distance = env.trajectory_distance  # (num_envs,)
    
    # 条件1：距离目标足够近
    distance_mask = distance < distance_threshold
    
    # 条件2：进度接近完成
    progress_mask = progress > progress_threshold
    
    # 条件3：速度足够慢
    box_lin_vel = env.scene[box_name].data.root_link_vel_w[:, :2]
    box_speed = torch.norm(box_lin_vel, dim=1)
    speed_mask = box_speed < speed_threshold
    
    # 组合所有条件（与 reward 函数完全一致）
    finished = distance_mask & progress_mask & speed_mask
    return finished
