from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_distance(
    env: ManagerBasedRLEnv,
    entity1: str | torch.Tensor,
    entity2: str | torch.Tensor,
    command_name: str | None = None,
) -> torch.Tensor:
    """
    Calculate the 2D Euclidean distance between two entities or positions.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        entity1 (str | torch.Tensor): Name of the first entity (e.g., 'robot', 'box_1') or a tensor of positions (num_envs, 2).
        entity2 (str | torch.Tensor): Name of the second entity or a tensor of positions (num_envs, 2).
        command_name (str | None): If entity2 is a command (e.g., goal pose), the name of the command.

    Returns:
        torch.Tensor: Euclidean distances, shape (num_envs,).
    """
    # Get position of entity1
    if isinstance(entity1, str):
        pos1 = env.scene[entity1].data.body_link_state_w[:, 0, :2]  # (num_envs, 2)
    else:
        pos1 = entity1[:, :2]  # Assume tensor input is (num_envs, 2)

    # Get position of entity2
    if isinstance(entity2, str):
        pos2 = env.scene[entity2].data.body_link_state_w[:, 0, :2]  # (num_envs, 2)
    elif isinstance(entity2, torch.Tensor):
        pos2 = entity2[:, :2]  # Direct tensor input
    elif command_name is not None:
        # Goal pose relative to box_1
        goal_pose_b = env.command_manager.get_command(command_name)  # (num_envs, 7)
        box_data = env.scene["box_1"].data.body_link_state_w[:, 0, :7]
        box_pos = box_data[:, :3]
        box_quat = box_data[:, 3:7]
        offset_pos_local = goal_pose_b[:, :3]
        offset_pos_world = quat_apply(box_quat, offset_pos_local)  # (num_envs, 3)
        pos2 = (box_pos + offset_pos_world)[:, :2]  # (num_envs, 2)
    else:
        raise ValueError("entity2 must be a string, tensor, or command_name must be provided.")

    # Calculate 2D Euclidean distance
    return torch.norm(pos1 - pos2, p=2, dim=-1)  # (num_envs,)

def box_reached_target_reward(
    env,
    command_name: str = "box_target",
    box_name: str = "box_1",
    threshold_pos: float = 0.05,
    threshold_rot: float = 0.1,
    speed_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Check if the box has reached the target position and orientation.
    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to track.
        box_name (str): The name of the box to check speed for.
        threshold_pos (float): The position error threshold to consider the box as reached.
        threshold_rot (float): The rotation error threshold to consider the box as reached.
        speed_threshold (float): Maximum allowed planar speed at the goal.
    Returns:
        torch.Tensor: A tensor indicating whether the box has reached the target (1.0)
                      or not (0.0) for each environment, shape (num_envs,).
    """
    # Get the box's relative target command (goal_pose_b: [dx, dy, dz, qw, qx, qy, qz] in box body coordinates)
    goal_pose_b = env.command_manager.get_command(command_name)  # (num_envs, 7)
    rel_pos = goal_pose_b[:, :3]
    rel_rot = goal_pose_b[:, 3:7]

    # Calculate the relative position error
    pos_error = torch.norm(rel_pos, dim=1)  # (num_envs,)

    # Calculate the relative rotation error (quaternion angle)
    dot_product = torch.sum(rel_rot * torch.tensor([1, 0, 0, 0], device=rel_rot.device), dim=1)
    dot_product = torch.abs(dot_product)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    rot_error = 2.0 * torch.acos(dot_product)  # (num_envs,)

    # Require low linear speed at the goal to prevent overshoot (mirror termination)
    box_lin_vel = env.scene[box_name].data.root_link_vel_w[:, :2]
    box_speed = torch.norm(box_lin_vel, dim=1)
    speed_ok = box_speed < speed_threshold

    # Check if the box has reached the target with low speed
    reached = (pos_error < threshold_pos) & speed_ok  # & (rot_error < threshold_rot)

    # Compute normalized time-based reward (encourage early success)
    remaining_steps = env.max_episode_length - env.episode_length_buf  # (num_envs,)
    reward_scale = remaining_steps / (env.max_episode_length + 1e-8)  # (num_envs,)
    # if reached.any():
    #     print(f"[DEBUG] Box reached target reward: {reached.float() * 2 * reward_scale}")
    return reached.float() * 2 * reward_scale  # (num_envs,)

def distance_to_target_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate and return the distance to the target, only apply if the robot has reached the box.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to track.

    Returns:
        torch.Tensor: The distance-based reward, shape (num_envs,)
    """
    reward = torch.zeros(env.num_envs, device=env.device)
    # Calculate distance from robot to target (box + offset)
    distance = get_distance(env, "robot", None, command_name=command_name)  # (num_envs,)
    
    # Reward scales inversely with distance
    reward_scale = 1.0 / (1 + 0.3 * torch.exp(1.5 * (distance * distance - 1.0))) / 1.42
    reward[env.reached_box_flags] = reward_scale[env.reached_box_flags]  # Only apply reward if the robot has reached the box
    # print(f"[DEBUG] Distance to target reward: {reward}")
    return reward

def distance_to_box_reward(
    env: ManagerBasedRLEnv,
    box_name: str,
    asset_name: str = "robot",
) -> torch.Tensor:
    """
    Calculate the reward based on the distance from the robot to the box. The closer the robot is to the box, the higher the reward.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        asset_name (str): The name of the robot asset.

    Returns:
        torch.Tensor: The distance-based reward, shape (num_envs,)
    """
    # Calculate distance from robot to box
    distance = get_distance(env, asset_name, box_name)  # (num_envs,)

    reward = 1.0 / (1 + 0.3 * torch.exp(1.5 * (distance * distance - 1.0))) / 1.42
    # print("[DEBUG] Distance to box reward:", reward)
    return reward 

def distance_decrease_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    robot_box_weight: float = 1.0,
    box_goal_weight: float = 2.5,
    scale: float = 10.0,
) -> torch.Tensor:
    """
    Calculate reward based on the decrease in distances: robot-to-box and box-to-goal.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        robot_box_weight (float): Weight for robot-to-box distance decrease.
        box_goal_weight (float): Weight for box-to-goal distance decrease.
        scale (float): Scaling factor for the combined reward.

    Returns:
        torch.Tensor: Reward based on distance decreases, shape (num_envs,).
    """
    # Calculate current distances
    current_robot_box_dist = get_distance(env, "robot", box_name)  # (num_envs,)
    
    goal_pos = env.goal_positions[:, :2]  # (num_envs, 2)
    robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]  # (num_envs, 2)
    current_box_goal_dist = torch.norm(robot_pos - goal_pos, dim=1)  # (num_envs,)

    # Calculate distance decreases (positive if distance reduced)
    decrease_robot_box = env._prev_robot_box_dist - current_robot_box_dist  # (num_envs,)
    decrease_box_goal = env._prev_box_goal_dist - current_box_goal_dist  # (num_envs,)
    # print(f"[DEBUG] Distance decrease: robot-box={decrease_robot_box}, box-goal={decrease_box_goal}")

    if not hasattr(env, "reached_box_flags"):
        env.reached_box_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Combine rewards with weights, only apply box-to-goal reward if robot reached box
    reward = torch.zeros(env.num_envs, device=env.device)

    reward[~env.reached_box_flags] += decrease_robot_box[~env.reached_box_flags] * robot_box_weight
    reward[env.reached_box_flags] += decrease_box_goal[env.reached_box_flags] * box_goal_weight
    reward *= scale

    # Update previous distances for next step
    env._prev_robot_box_dist = current_robot_box_dist.clone()
    env._prev_box_goal_dist = current_box_goal_dist.clone()

    # print(f"[DEBUG] Distance decrease reward: {reward}")
    return reward

def box_movement_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    speed_threshold: float = 0.01,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """
    Calculate the reward based on the box's linear velocity.

    The reward is proportional to the box's speed if it exceeds a threshold,
    and is normalized by episode length for training stability.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        speed_threshold (float): Minimum speed considered as movement.
        normalization_scale (float): The speed mapped to reward=1.0.
        max_reward (float): Maximum reward before normalization.

    Returns:
        torch.Tensor: The movement-based reward, shape (num_envs,)
    """
    # Get the box entity from the scene
    box = env.scene[box_name]

    # Extract linear velocity of the box (num_envs, 2)
    lin_vel = box.data.root_link_vel_w[:, :2]

    # Compute speed magnitude
    speed = torch.norm(lin_vel, dim=-1)  # (num_envs,)

    # Only reward movement above threshold
    active_movement = torch.where(speed > speed_threshold, speed, torch.zeros_like(speed))

    # Clamp reward to maximum value
    reward = torch.clamp(active_movement, max=max_reward) 

    # print("[DEBUG] Box movement reward:", reward, )
    # Normalize by episode length 
    return reward 


def slowdown_near_target_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    command_name: str = "box_target",
    slowdown_distance: float = 0.35,
    speed_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Encourage the box to slow down as it approaches the goal.

    Provides positive reward when the box is close to the target and moving
    slower than the threshold, and a penalty when the box is close but still moving fast.
    """
    # Distance from box to goal
    distance = get_distance(env, box_name, None, command_name=command_name)

    # Linear speed of the box in the plane
    box_lin_vel = env.scene[box_name].data.root_link_vel_w[:, :2]
    box_speed = torch.norm(box_lin_vel, dim=1)

    # Only apply shaping close to the goal
    proximity_weight = torch.clamp((slowdown_distance - distance) / slowdown_distance, min=0.0, max=1.0)

    safe_speed = max(speed_threshold, 1e-3)
    speed_weight = torch.clamp((safe_speed - box_speed) / safe_speed, min=-1.0, max=1.0)
    return proximity_weight * speed_weight


def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Calculate the oscillation penalty.

    This function penalizes the robot for oscillatory movements by comparing the difference
    in consecutive actions. If the difference exceeds a threshold, a squared penalty is applied.
    """
    # Accessing the robot's actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculating differences between consecutive actions
    linear_diff = action[:, 0] - prev_action[:, 0]
    angular_diff = action[:, 1] - prev_action[:, 1]

    angular_penalty = torch.where(
        angular_diff*3 > 0.05, torch.square(angular_diff*3), 0.0)
    linear_penalty = torch.where(
        linear_diff*3 > 0.05, torch.square(linear_diff*3), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)

    # print("[DEBUG] Oscillation penalty:", (angular_penalty + linear_penalty) )

    return (angular_penalty + linear_penalty) 

def out_of_bounds_penalty(
    env: ManagerBasedRLEnv,
    radius: float = 2.0
) -> torch.Tensor:
    """
    Calculate the out-of-bounds penalty for circular boundary.

    This function penalizes the robot for being outside a defined circular area.
    The penalty is proportional to the distance from the circular boundary.
    """
    # Get the robot's position
    asset = env.scene["robot"]
    position = asset.data.root_link_state_w[:, :2]  # (num_envs, 2)
    env_origins = env.scene.env_origins
    
    # Adjust position to the environment origin
    position = position - env_origins[:, :2]  # (num_envs, 2)

    # Calculate distance from origin (center of circular boundary)
    distance_from_center = torch.norm(position, dim=1)  # (num_envs,)

    # Calculate penalty for being outside the circular boundary
    # If distance > radius, apply penalty proportional to excess distance
    excess_distance = torch.clamp(distance_from_center - radius, min=0.0)  # (num_envs,)
    
    # Square the penalty for stronger effect (similar to original function)
    penalty = excess_distance * excess_distance
    
    # Normalize the penalty 
    penalty = torch.clamp(penalty, max=20.0)  # Cap the penalty to avoid extreme values
    
    return penalty

def angle_to_box_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    asset_name: str = "robot",
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Reward based on the alignment between the robot's x-axis and the direction to the box,
    only when the robot is farther than a threshold distance from the box.
    """
    # Check if the box has been reached
    if not hasattr(env, "reached_box_flags"):
        env.reached_box_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    box = env.scene[box_name]
    robot = env.scene[asset_name]

    # Positions in world frame
    box_pos = box.data.body_link_state_w[:, 0, :2]  # (B, 2)
    robot_pos = robot.data.body_link_state_w[:, 0, :2]  # (B, 2)
    direction_world = box_pos - robot_pos  # (B, 2)

    # Robot orientation (quaternion to yaw)
    robot_quat = robot.data.root_state_w[:, 3:7]  # (B, 4)
    siny_cosp = 2 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2])
    cosy_cosp = 1 - 2 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)  # (B,)

    # Rotate direction vector to robot frame (rotate by -yaw)
    cos_yaw = torch.cos(-robot_yaw)
    sin_yaw = torch.sin(-robot_yaw)
    local_x = direction_world[:, 0] * cos_yaw - direction_world[:, 1] * sin_yaw
    local_y = direction_world[:, 0] * sin_yaw + direction_world[:, 1] * cos_yaw

    # Angle in robot frame: ideally should be close to 0
    angle = torch.atan2(local_y, local_x)  # (B,)
    angle_error = torch.abs(angle) / torch.pi  # (B,)

    # Reward: the smaller the angle error, the higher the reward
    reward = 1 - 3 * angle_error ** 2 + 2 * angle_error ** 3  # smooth reward 1-3x^{2}+2x^{3}

    # Suppress angle reward when too close
    distance = torch.norm(direction_world, dim=-1)
    reward = torch.where(distance > distance_threshold, reward, torch.zeros_like(reward))
    # print("[DEBUG] Angle to box reward:", reward)
    reward[env.reached_box_flags] = 0.0  # No angle reward if box already reached

    return reward 

def reached_box_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    distance_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Calculate and return the reward for reaching the box.
    """
    reward = torch.zeros(env.num_envs, device=env.device)

    box = env.scene[box_name]
    robot = env.scene["robot"]
    box_position = box.data.body_link_state_w[:, 0, :2]  # Get box position in world frame
    robot_position = robot.data.body_link_state_w[:, 0, :2]  # Get robot position in world frame
    distance = torch.norm(box_position - robot_position, dim=1)  # Calculate distance to box

    remaining_steps = env.max_episode_length - env.episode_length_buf
    # Normalize the reward based on the remaining steps
    reward_scale = remaining_steps  / (env.max_episode_length + 1e-8)

    if not hasattr(env, "reached_box_flags"):
        env.reached_box_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    newly_reached = (distance < distance_threshold) & (~env.reached_box_flags)
    # Update the reached box flags
    env.reached_box_flags = env.reached_box_flags | newly_reached
    # Apply the reward only to first time reaching the box
    reward[newly_reached] = 2 * reward_scale[newly_reached]

    return reward

def trajectory_distance_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    distance_threshold: float = 0.2,
    reward_scale: float = 0.1,
) -> torch.Tensor:
    """
    Reward based on the distance from the box to the trajectory.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        distance_threshold (float): The distance threshold for maximum reward.

    Returns:
        torch.Tensor: The trajectory distance-based reward, shape (num_envs,).
    """
    if not hasattr(env, 'trajectory_distance'):
        return torch.zeros(env.num_envs, device=env.device)
    
    distance = env.trajectory_distance  # Distance from box to trajectory

    # Reward scales inversely with distance
    reward = torch.exp( -distance**2 / (2 * distance_threshold**2) )
    reward = (reward - 0.5) * 2.0 * reward_scale  # Scale to [0, reward_scale]
    reward[~env.reached_box_flags] = 0.0  # Only apply reward if the robot has reached the box
    # print("[DEBUG] Trajectory distance reward:", reward)
    return reward

def trajectory_progress_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    reward_amount: float = 1.0,
) -> torch.Tensor:
    """
    Reward based on the progress along the trajectory.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
    Returns:
        torch.Tensor: The trajectory progress-based reward, shape (num_envs,).
    """
    if not hasattr(env, 'trajectory_progress'):
        return torch.zeros(env.num_envs, device=env.device)
    
    progress = env.trajectory_progress  # Progress along the trajectory [0, 1]

    if not hasattr(env, "_prev_trajectory_progress"):
        env._prev_trajectory_progress = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    progress_delta = progress - env._prev_trajectory_progress
    
    # deal with looping progress
    is_loop = progress_delta < -0.5
    progress_delta[is_loop] += 1.0
    
    progress_reward = torch.where(
        progress_delta > 0,
        progress_delta * 50,  # forward progress
        progress_delta * 100   # backward progress
    )
    
    # env._prev_trajectory_progress = progress.clone()
    
    progress_reward[~env.reached_box_flags] = 0.0  # Only apply reward if the robot has reached the box
    return progress_reward * reward_amount

def trajectory_velocity_alignment_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    velocity_alignment_weight: float = 0.005,
) -> torch.Tensor:
    """
    Reward based on the alignment between the box's velocity and the trajectory's direction.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        velocity_alignment_weight (float): Weight for the velocity alignment reward.
    Returns:
        torch.Tensor: The trajectory velocity alignment-based reward, shape (num_envs,).
    """
    if not hasattr(env, 'trajectory_closest_points'):
        return torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, '_prev_closest_points'):
        return torch.zeros(env.num_envs, device=env.device)
        
    box_vel = env.scene[box_name].data.root_link_vel_w[:, :2]
    box_speed = torch.norm(box_vel, dim=-1, keepdim=True) + 1e-6

    closest_idx = env.closest_point_index  # Current closest trajectory point index
    lookahead_steps = 3  # Look ahead 3 points
    target_idx = torch.clamp(closest_idx + lookahead_steps, max=len(env.current_trajectories[0])-1)
    env_indices = torch.arange(env.num_envs, device=env.device)
    target_point = env.current_trajectories[env_indices, target_idx]  # (num_envs, 3)

    traj_dir = target_point[:, :2] - env.trajectory_closest_points[:, :2]
    traj_dir = traj_dir / (torch.norm(traj_dir, dim=-1, keepdim=True) + 1e-6)
    
    # Velocity direction
    vel_dir = box_vel / box_speed
    
    # Alignment
    alignment = torch.sum(vel_dir * traj_dir, dim=-1)
    velocity_reward = torch.clamp(alignment, min=-0.5, max=1.0)
    velocity_alignment_reward = velocity_alignment_weight * velocity_reward
    velocity_alignment_reward[~env.reached_box_flags] = 0.0  # Only apply reward if the robot has reached the box
    
    # env._prev_closest_points = env.trajectory_closest_points.clone()

    return velocity_alignment_reward

# def trajectory_off_track_penalty(
#     env: ManagerBasedRLEnv,
#     distance_threshold: float = 0.2,
#     off_track_penalty: float = 0.5,
# ) -> torch.Tensor:
#     """
#     Penalty for being off the trajectory.

#     Args:
#         env (ManagerBasedRLEnv): The environment instance.
#         distance_threshold (float): The distance threshold for off-track penalty.
#         off_track_penalty (float): The penalty value when off-track.

#     Returns:
#         torch.Tensor: The trajectory off-track penalty, shape (num_envs,).
#     """
#     if not hasattr(env, 'trajectory_distance'):
#         return torch.zeros(env.num_envs, device=env.device)
    
#     distance = env.trajectory_distance
#     off_track_mask = distance > (distance_threshold)
#     off_track_penalty_value = off_track_penalty * (distance - distance_threshold)
#     off_track_penalty = torch.zeros(env.num_envs, device=env.device)
#     off_track_penalty[off_track_mask] = off_track_penalty_value[off_track_mask]
#     off_track_penalty[~env.reached_box_flags] = 0.0  # Only apply penalty if the robot has reached the box
#     return off_track_penalty

def trajectory_backward_penalty(
    env: ManagerBasedRLEnv,
    backward_penalty_value: float = 1.0,
) -> torch.Tensor:
    """
    Penalty for moving backward along the trajectory.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        backward_penalty (float): The penalty value when moving backward.

    Returns:
        torch.Tensor: The trajectory backward penalty, shape (num_envs,).
    """
    if not hasattr(env, 'trajectory_progress'):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, '_prev_trajectory_progress'):
        return torch.zeros(env.num_envs, device=env.device)
    
    progress_delta = env.trajectory_progress - env._prev_trajectory_progress
    
    backward_mask = progress_delta < -0.02  # Small fluctuations are not considered backward
    backward_mask = backward_mask & ~(progress_delta < -0.5)  # Exclude looping cases

    backward_penalty = torch.zeros(env.num_envs, device=env.device)
    backward_penalty[backward_mask] = backward_penalty_value
    backward_penalty[~env.reached_box_flags] = 0.0  # Only apply penalty if the robot has reached the box
    return backward_penalty

def trajectory_jump_penalty(
    env: ManagerBasedRLEnv,
    jump_threshold: float = 0.1,
    distance_jump_penalty: float = 0.2,
) -> torch.Tensor:
    """
    Penalty for sudden jumps in trajectory distance.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        jump_threshold (float): The threshold for detecting jumps.
        distance_jump_penalty: float = -0.2,
    Returns:
        torch.Tensor: The trajectory jump penalty, shape (num_envs,).
    """

    if not hasattr(env, 'trajectory_distance'):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, '_prev_trajectory_distance'):
        return torch.zeros(env.num_envs, device=env.device)
    
    distance_change = torch.abs(env.trajectory_distance - env._prev_trajectory_distance)
    jump_mask = distance_change > jump_threshold  # Threshold for jump detection
    
    jump_penalty = torch.zeros(env.num_envs, device=env.device)
    jump_penalty[jump_mask] = distance_change[jump_mask] * distance_jump_penalty
    jump_penalty[~env.reached_box_flags] = 0.0  # Only apply penalty if the robot has reached the box
    
    # env._prev_trajectory_distance = env.trajectory_distance.clone()
    return jump_penalty

def trajectory_following_reward_milestone(
    env: ManagerBasedRLEnv,
    milestone: float = 0.5,              # 配置这个值：0.3、0.5、0.7、0.9...
    milestone_reward: float = 5.0,       # 这次到达想发多少奖励
    distance_threshold: float = 0.2,     # 离轨迹太远不算
) -> torch.Tensor:
    """
    单个里程碑奖励函数 —— 干净、快速、绝对只发一次、无法刷分
    使用方式：在 config 中写多个这样的 term，例如：
        - name: milestone_30
          weight: 1.0
          params:
            milestone: 0.30
            milestone_reward: 3.0
        - name: milestone_50
          weight: 1.0
          params:
            milestone: 0.50
            milestone_reward: 8.0
    """
    # ---------- 1. 初始化永久 flag（每个 env 每个 term 独立）----------
    flag_name = f"_milestone_{milestone}_flag"
    if not hasattr(env, flag_name):
        env.__dict__[flag_name] = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    # ---------- 2. 必要条件检查 ----------
    if not hasattr(env, 'trajectory_progress') or not hasattr(env, 'trajectory_distance'):
        return torch.zeros(env.num_envs, device=env.device)

    progress = env.trajectory_progress          # [num_envs]
    distance = env.trajectory_distance          # [num_envs]

    # 是否已经先碰到箱子（防止还没接触箱子就乱算进度）
    reached_box = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "reached_box_flags"):
        reached_box = env.reached_box_flags
    
    # ---------- 3. 核心判断：是否“曾经达到过”且“还没发过奖”且“现在在轨迹上” ----------
    has_reached = progress >= milestone
    not_off_track = distance <= distance_threshold
    not_rewarded_yet = ~getattr(env, flag_name)

    should_reward = has_reached & not_rewarded_yet & not_off_track & reached_box

    # ---------- 4. 发奖 + 永久锁死 ----------
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[should_reward] = milestone_reward

    # 永久锁死（一旦达到过就锁死，哪怕后面掉轨、后退都不解锁）
    getattr(env, flag_name)[:] = getattr(env, flag_name) | has_reached
    
    return reward

def trajectory_progress_finish_reward(
    env: ManagerBasedRLEnv,
    reward_amount: float = 0.2,
) -> torch.Tensor:
    """
    Reward for finishing the trajectory by reaching the target.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        reward_amount (float): Reward amount for finishing the trajectory.

    Returns:
        torch.Tensor: Reward for finishing the trajectory, shape (num_envs,).
    """
    reward = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, 'flag_trajectory_finished'):
        env.flag_trajectory_finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    if hasattr(env, 'num_finish_goals'):
        num_finish_goals = env.num_finish_goals #(num_envs,)
        total_goals = env.total_goals # int
        finished_mask = num_finish_goals == total_goals # (num_envs,)
        just_finished_mask = finished_mask & (~env.flag_trajectory_finished)
        
        # 计算时间缩放奖励
        remaining_steps = env.max_episode_length - env.episode_length_buf
        reward_scale = remaining_steps / (env.max_episode_length + 1e-8)
        reward[just_finished_mask] = reward_amount * reward_scale[just_finished_mask]
        
        env.flag_trajectory_finished |= just_finished_mask

    return reward

def slow_down_near_goal_reward(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
    distance_threshold: float = 0.2,
    speed_threshold: float = 0.2,
    reward_amount: float = 0.5,
) -> torch.Tensor:
    """
    Encourage the box to slow down as it approaches the goal.
    Provides positive reward when the box is close to the target and moving
    Pargs:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        distance_threshold (float): The distance threshold for slowing down.
        speed_threshold (float): The speed threshold for slowing down.
        reward_amount (float): The maximum reward amount.
    Returns:
        torch.Tensor: The slow down near goal reward, shape (num_envs,).
    """
    
    # distance from box to goal
    box_pos = env.scene[box_name].data.body_link_state_w[:, 0, :2]  # (num_envs, 2)
    goal_pos = env.goal_positions[:, :2]  # (num_envs, 2)
    distance = torch.norm(box_pos - goal_pos, dim=1)
    
    # linear speed of the box in the plane
    # Linear speed of the box in the plane
    box_lin_vel = env.scene[box_name].data.root_link_vel_w[:, :2]
    box_speed = torch.norm(box_lin_vel, dim=1)

    # Only apply shaping close to the goal
    proximity_weight = torch.clamp((distance_threshold - distance) / distance_threshold, min=0.0, max=1.0)

    safe_speed = max(speed_threshold, 1e-3)
    speed_weight = torch.clamp((safe_speed - box_speed) / safe_speed, min=-1.0, max=1.0)
    return proximity_weight * speed_weight * reward_amount

def finish_goal_reward(
    env: ManagerBasedRLEnv,
    reward_amount: float = 1.0,
    distance_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward for finishing each goal along the trajectory.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        reward_amount (float): Reward amount for finishing each goal.

    Returns:
        torch.Tensor: Reward for finishing goals, shape (num_envs,).
    """
    
    reward = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, 'flag_reach_goal'):
        env.flag_reach_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)    # (num_envs,)
        return reward
    
    box = env.scene["box_1"]
    box_pos = box.data.body_link_state_w[:, 0, :3]  # (num_envs, 3)
    box_vel = box.data.root_link_vel_w[:, :3]   # (num_envs, 3)
    box_speed = torch.norm(box_vel, dim=1)  #(num_envs,)

    # arrive the goal
    goal_pos = env.goal_positions  # (num_envs, 3)
    pos_error = torch.norm(box_pos - goal_pos, dim=1)  # (num_envs,)
    reached_pos = pos_error < distance_threshold
    
    # velocity check
    reached_speed = box_speed < velocity_threshold
    
    # only reward once
    reached_goal = reached_pos & reached_speed
    reward[reached_goal] = reward_amount / env.total_goals # distribute reward over all goals
    
    env.flag_reach_goal = reached_goal # update the flag, update the goal in step function

    return reward