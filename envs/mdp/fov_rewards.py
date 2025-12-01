"""
FOV-based reward functions for Isaac Lab environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
import torch
from envs.mdp.fov_observations import is_in_fov_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fov_exploration_reward(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    box_names: List[str] = None,
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Reward for keeping objects in field of view.
    
    Args:
        env: The RL environment
        robot_name: Name of the robot asset
        box_names: List of box names to track
        h_fov: Horizontal field of view
        v_fov: Vertical field of view
        focal_length: Camera focal length
        max_detection_distance: Maximum detection distance
        is_deg: Whether FOV angles are in degrees
        
    Returns:
        Reward tensor for FOV exploration
    """
    if box_names is None:
        box_names = ["box_1"]
    
    robot = env.scene[robot_name]
    robot_pos = robot.data.root_state_w[:, :3]
    robot_quat = robot.data.root_state_w[:, 3:7]
    
    # Collect box positions
    box_positions = []
    for box_name in box_names:
        box = env.scene[box_name]
        box_pos = box.data.root_state_w[:, :3]
        box_positions.append(box_pos)
    
    if len(box_names) > 1:
        box_positions = torch.stack(box_positions, dim=1)
    else:
        box_positions = box_positions[0].unsqueeze(1)
    
    # Check FOV visibility
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    # Reward for objects in FOV
    fov_reward = is_in_fov.float().sum(dim=-1)  # Sum over objects
    # print(f"fov_reward: {fov_reward.unsqueeze(-1)}")
    
    return fov_reward


def fov_tracking_reward(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    box_names: List[str] = None,
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    center_weight: float = 2.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Reward for keeping objects centered in field of view.
    Higher reward when objects are closer to the center of FOV.
    """
    if box_names is None:
        box_names = ["box_1"]
    
    robot = env.scene[robot_name]
    robot_pos = robot.data.root_state_w[:, :3]
    robot_quat = robot.data.root_state_w[:, 3:7]
    
    # Collect box positions
    box_positions = []
    for box_name in box_names:
        box = env.scene[box_name]
        box_pos = box.data.root_state_w[:, :3]
        box_positions.append(box_pos)
    
    if len(box_names) > 1:
        box_positions = torch.stack(box_positions, dim=1)
    else:
        box_positions = box_positions[0].unsqueeze(1)
    
    # Transform to robot frame to calculate angular deviation
    from envs.mdp.fov_observations import coordinate_transform
    
    batch_size = robot_pos.shape[0]
    num_objects = box_positions.shape[1]
    
    # Expand robot pose for each object
    robot_pos_expanded = robot_pos.unsqueeze(1).expand(-1, num_objects, -1)
    robot_quat_expanded = robot_quat.unsqueeze(1).expand(-1, num_objects, -1)
    
    # Transform to robot coordinates
    obj_pos_robot = coordinate_transform(
        robot_pos_expanded.reshape(-1, 3),
        robot_quat_expanded.reshape(-1, 4),
        box_positions.reshape(-1, 3)
    ).reshape(batch_size, num_objects, 3)
    
    # Check FOV visibility first
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    # Calculate angular deviation from center (robot's x-axis)
    # Convert to drone coordinate system: (x,y,z) -> (y,z,x)
    obj_pos_drone = obj_pos_robot[..., [1, 2, 0]]
    
    # Calculate angles in image plane
    horizontal_angle = torch.atan2(torch.abs(obj_pos_drone[..., 0]), obj_pos_drone[..., 2])
    vertical_angle = torch.atan2(torch.abs(obj_pos_drone[..., 1]), obj_pos_drone[..., 2])
    
    # Normalize by half FOV
    import math
    if is_deg:
        h_fov_rad = h_fov * math.pi / 180.0
        v_fov_rad = v_fov * math.pi / 180.0
    else:
        h_fov_rad = h_fov
        v_fov_rad = v_fov
    
    h_norm_angle = horizontal_angle / (h_fov_rad / 2.0)
    v_norm_angle = vertical_angle / (v_fov_rad / 2.0)
    
    # Center reward: higher when closer to center
    center_reward = center_weight * (1.0 - torch.sqrt(h_norm_angle**2 + v_norm_angle**2))
    center_reward = torch.clamp(center_reward, min=0.0)
    
    # Apply only to visible objects
    center_reward = center_reward * is_in_fov.float()
    
    return center_reward.sum(dim=-1).unsqueeze(-1)


def fov_lost_target_penalty(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    box_names: List[str] = None,
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Penalty for losing track of important objects.
    """
    if box_names is None:
        box_names = ["box_1"]
    
    robot = env.scene[robot_name]
    robot_pos = robot.data.root_state_w[:, :3]
    robot_quat = robot.data.root_state_w[:, 3:7]
    
    # Collect box positions
    box_positions = []
    for box_name in box_names:
        box = env.scene[box_name]
        box_pos = box.data.root_state_w[:, :3]
        box_positions.append(box_pos)
    
    if len(box_names) > 1:
        box_positions = torch.stack(box_positions, dim=1)
    else:
        box_positions = box_positions[0].unsqueeze(1)
    
    # Check FOV visibility
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    # Penalty for objects that are in range but not in FOV
    lost_target_penalty = is_in_dist.float() * (~is_in_fov).float()
    
    return -lost_target_penalty.sum(dim=-1).unsqueeze(-1)


def fov_distance_adaptive_reward(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    box_names: List[str] = None,
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    optimal_distance: float = 2.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Adaptive reward based on distance to objects in FOV.
    Encourages maintaining optimal viewing distance.
    """
    if box_names is None:
        box_names = ["box_1"]
    
    robot = env.scene[robot_name]
    robot_pos = robot.data.root_state_w[:, :3]
    robot_quat = robot.data.root_state_w[:, 3:7]
    
    # Collect box positions
    box_positions = []
    for box_name in box_names:
        box = env.scene[box_name]
        box_pos = box.data.root_state_w[:, :3]
        box_positions.append(box_pos)
    
    if len(box_names) > 1:
        box_positions = torch.stack(box_positions, dim=1)
    else:
        box_positions = box_positions[0].unsqueeze(1)
    
    # Calculate distances
    distances = torch.norm(box_positions - robot_pos.unsqueeze(1), dim=-1)
    
    # Check FOV visibility
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    # Distance reward: higher when closer to optimal distance
    distance_error = torch.abs(distances - optimal_distance)
    distance_reward = torch.exp(-distance_error / optimal_distance)
    
    # Apply only to visible objects
    distance_reward = distance_reward * is_in_fov.float()
    
    return distance_reward.sum(dim=-1).unsqueeze(-1)
    