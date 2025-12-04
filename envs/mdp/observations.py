from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import yaw_quat, quat_apply

from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def distance_robot_to_goal(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Calculate the distance from the robot to the goal position.
    """
    # world position of goal
    goal_pos = env.goal_position[:, :2] # (num_envs, 2)

    # Get robot position
    robot_pos = env.scene["robot"].data.body_link_state_w[:, 0, :2]
    goal_distance = torch.norm(goal_pos - robot_pos, dim=-1)  # Calculate Euclidean distance from the robot
    return goal_distance.unsqueeze(1)  # Return as (num_envs, 1) tensor

def angle_robot_to_goal(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Calculate the angle between the line connecting the robot and the goal 
    and the x-axis of the robot, in the robot's local frame.
    The goal is defined relative to box_1's local frame and converted to world coordinates.
    """

    # Final world position of goal
    goal_pos = env.goal_position  # (B, 3)

    # Get robot position
    robot_pos = env.scene["robot"].data.body_link_state_w[:, 0, :3]
    direction_world = goal_pos[:, :2] - robot_pos[:, :2]  # xy-plane direction

    # Get robot yaw from quaternion
    robot_quat = env.scene["robot"].data.root_state_w[:, 3:7]
    siny_cosp = 2 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2])
    cosy_cosp = 1 - 2 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Transform direction to robot local frame
    cos_yaw = torch.cos(-robot_yaw)
    sin_yaw = torch.sin(-robot_yaw)
    local_x = direction_world[:, 0] * cos_yaw - direction_world[:, 1] * sin_yaw
    local_y = direction_world[:, 0] * sin_yaw + direction_world[:, 1] * cos_yaw

    # Return angle in robot frame
    angle_local = torch.atan2(local_y, local_x)
    return angle_local.unsqueeze(1)

def distance_to_box(
    env: ManagerBasedRLEnv,
    box_name: str,
    asset_name: str = "robot",
) -> torch.Tensor:
    """
    Calculate the distance from the robot to the boxes.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        box_name (str): The name of the box entity.
        asset_name (str): The name of the robot asset.

    Returns:
        torch.Tensor: The distances from the robot to the box.
    """
    box = env.scene[box_name]
    asset = env.scene[asset_name]
    box_position = box.data.body_link_state_w[:, 0, :2]  # Get box position in world frame
    robot_position = asset.data.body_link_state_w[:, 0, :2]  # Get robot position in world frame
    return torch.norm((box_position - robot_position).unsqueeze(1), dim=-1)

def angle_to_box(
    env: ManagerBasedRLEnv,
    box_name: str,
    asset_name: str = "robot",
) -> torch.Tensor:
    """
    Calculate the angle between the line connecting the robot and the box and the x-axis of the robot.
    This is done in the robot's local frame.
    """
    box = env.scene[box_name]
    robot = env.scene[asset_name]

    # World positions
    box_pos = box.data.body_link_state_w[:, 0, :2]
    robot_pos = robot.data.body_link_state_w[:, 0, :2]
    direction_world = box_pos - robot_pos  # (B, 2)

    # Robot heading: get yaw (rotation around z)
    robot_quat = robot.data.root_state_w[:, 3:7]  # quaternion (B, 4)
    # Convert quaternion to yaw angle (robot heading in world frame)
    siny_cosp = 2 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2])
    cosy_cosp = 1 - 2 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    robot_yaw = torch.atan2(siny_cosp, cosy_cosp)  # (B,)

    # Rotate the direction vector into robot's local frame (rotate by -yaw)
    cos_yaw = torch.cos(-robot_yaw)
    sin_yaw = torch.sin(-robot_yaw)
    local_x = direction_world[:, 0] * cos_yaw - direction_world[:, 1] * sin_yaw
    local_y = direction_world[:, 0] * sin_yaw + direction_world[:, 1] * cos_yaw

    # Final angle in robot frame
    angle_local = torch.atan2(local_y, local_x)  # (B,)

    return angle_local.unsqueeze(1)

def get_camera_observation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get the camera observation from the environment.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        sensor_cfg (SceneEntityCfg): The configuration of the camera sensor.

    Returns:
        torch.Tensor: The camera observation as a tensor.
    """
    camera_sensor = env.get_sensor(sensor_cfg.prim_path)
    if not isinstance(camera_sensor, CameraCfg):
        raise TypeError(f"Expected CameraCfg, got {type(camera_sensor)}")
    
    return camera_sensor.get_observation().to_tensor()

def get_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    flatten: bool = True,  
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.
        flatten: Whether to flatten the image to 1D vector for concatenation with other observations.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    # Flatten image for concatenation with other observation terms
    if flatten:
        # Keep batch dimension, flatten remaining dimensions
        batch_size = images.shape[0]
        images = images.view(batch_size, -1)
    
    return images.clone()
        
def get_camera_rgbd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Get RGB-D images from camera sensor for student network.
    
    Args:
        env: The environment instance
        sensor_cfg: Camera sensor configuration
        normalize: Whether to normalize the images
        
    Returns:
        RGB-D images [num_envs, H, W, 4] (not flattened)
    """
    # Get camera sensor
    sensor: TiledCamera | Camera = env.scene.sensors[sensor_cfg.name]
    
    # Get RGB and depth data
    rgb_images = sensor.data.output["rgb"]  # [num_envs, H, W, 3]
    depth_images = sensor.data.output["distance_to_image_plane"]  # [num_envs, H, W, 1] or [num_envs, H, W]
    
    # Ensure depth has correct shape
    if depth_images.dim() == 3:  # [num_envs, H, W]
        depth_images = depth_images.unsqueeze(-1)  # [num_envs, H, W, 1]
    
    # Normalize if requested
    if normalize:
        # RGB: scale to [0, 1]
        rgb_images = rgb_images.float() / 255.0
        
        # Depth: clamp and normalize
        max_depth = 10.0  # 10 meters max
        depth_images = torch.clamp(depth_images, 0, max_depth) / max_depth
    
    # Concatenate RGB and depth
    rgbd_images = torch.cat([rgb_images, depth_images], dim=-1)  # [num_envs, H, W, 4]
    # print(f"RGB-D image shape: {rgbd_images.shape}")
    
    return rgbd_images

def get_trajectory_history_and_future(
    env: ManagerBasedRLEnv,
    past_steps: int = 10,
    future_steps: int = 30,
    flatten: bool = True,
) -> torch.Tensor:
    """
    Get trajectory history and future points for each env.
    
    Args:
        env: The environment instance
        past_steps: Number of past steps to include
        future_steps: Number of future steps to include
    Returns:
        Tensor of shape (B, total, 3) where total = past_steps + future_steps + (1 if include_current else 0)
    """
    B = env.num_envs
    device = env.device
    total = past_steps + future_steps + 1
    if not hasattr(env, 'current_trajectories') or env.current_trajectories is None:
        # If no trajectories available, return zeros
        if flatten:
            return torch.zeros((B, total * 2), device=device)
        else:
            return torch.zeros((B, total, 3), device=device)
    
    if not hasattr(env, 'trajectory_generator'):
        # If no trajectory generator, return zeros
        if flatten:
            return torch.zeros((B, total * 2), device=device)
        else:
            return torch.zeros((B, total, 3), device=device)

    # get current closest point indices on trajectory
    _, closest_indices, _, _ = env.trajectory_generator.get_closest_point_on_trajectory(
        box_pos=env.scene["box_1"].data.root_pos_w[:, :3],
        trajectories=env.current_trajectories
    )  # (B,)
    
    # Generate target index sequence (relative offsets)
    offsets = torch.arange(
        -past_steps,
        future_steps + 1,
        device=device
    )  # e.g., [-10, -9, ..., -1, 0, 1, ..., 30]

    # Broadcast to get absolute indices for each env: (B, total)
    indices = closest_indices.unsqueeze(1) + offsets.unsqueeze(0)  # (B, total)

    # Get lengths of each trajectory for clamping
    num_points = len(env.current_trajectories[0])  # should be the same for all envs if using same cfg when training
    traj_lengths = torch.full(
        (B,), 
        num_points, 
        device=device, 
        dtype=torch.long
    )  # (B,)
    # should be the same for all envs if using same cfg when training

    # Clamp indices to valid range [0, L_i-1]
    indices_clamped = torch.clamp(indices, min=0, max=num_points - 1)

    # Construct batch indices for gather
    batch_idx = torch.arange(B, device=device)

    # Use index_select + scatter to implement batch indexing for List[Tensor]
    # First concatenate all trajectories into one big tensor + offsets
    traj_cat = env.current_trajectories  # (total_points_all_envs, 3)
    
    indices_expanded = indices_clamped.unsqueeze(-1).expand(-1, -1, 3)  # (B, total, 3)
    # Actually gather points
    points = torch.gather(traj_cat, dim=1, index=indices_expanded)  # (B, total, 3)

    # replicate padding (vectorized!)
    # Construct mask: which indices are truly valid (not changed by clamp)
    valid_mask = (indices >= 0) & (indices < traj_lengths.unsqueeze(1))

    # Find the first and last valid points for each env
    first_valid = valid_mask.long().argmax(dim=1)   # (B,)
    last_valid = total - 1 - valid_mask.flip(1).long().argmax(dim=1)  # (B,)

    # Use the first valid point to fill the front, and the last valid point to fill the back
    first_points = points[batch_idx, first_valid]
    last_points  = points[batch_idx, last_valid]

    # Construct padding mask
    fill_front = torch.arange(total, device=device).unsqueeze(0) < first_valid.unsqueeze(1)
    fill_back  = torch.arange(total, device=device).unsqueeze(0) > last_valid.unsqueeze(1)

    # Final padding
    result = points.clone()
    result = torch.where(fill_front.unsqueeze(-1), first_points.unsqueeze(1), result)
    result = torch.where(fill_back.unsqueeze(-1),  last_points.unsqueeze(1),  result)
    
    # 转化为局部坐标系
    robot_pos = env.scene["robot"].data.root_pos_w[:, :3]  # (B, 3)
    robot_quat = env.scene["robot"].data.root_quat_w  # (B, 4)
    
    robot_pos_expanded = robot_pos.unsqueeze(1).expand(-1, total, -1)  # (B, total, 3)
    robot_quat_expanded = robot_quat.unsqueeze(1).expand(-1, total, -1)  # (B, total, 4)
    
    local_points = coordinate_transform(
        sys_pos=robot_pos_expanded.reshape(-1, 3),
        sys_quat=robot_quat_expanded.reshape(-1, 4),
        obj_pos=result.reshape(-1, 3)
    ).reshape(B, total, 3)  # (B, total, 3)
    
    local_points_xy = local_points[:, :, :2]    # (B, total, 2)

    if flatten:
        return local_points_xy.reshape(B, -1)  # (B, total*2)
    else:
        return local_points_xy  # (B, total, 2)

def get_goal_observation(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """
    Get the goal observation tensor for the current target position relative to the box.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
    Returns:
        torch.Tensor: The goal observation tensor, shape (num_envs, 2).
    """
    if not hasattr(env, 'current_trajectories') or env.current_trajectories is None:
        # If no trajectories available, return zeros
        return torch.zeros((env.num_envs, 2), device=env.device)

    goal_pos = env.current_trajectories[:, -1, :3]  # (num_envs, 3)
    
    robot_pos = env.scene["robot"].data.body_link_state_w[:, 0, :3]  # (num_envs, 3)
    robot_quat = env.scene["robot"].data.root_state_w[:, 3:7]  # (num_envs, 4)
    
    # Transform goal position to robot local frame
    goal_pos_local = coordinate_transform(
        sys_pos=robot_pos,
        sys_quat=robot_quat,
        obj_pos=goal_pos
    )  # (num_envs, 3)

    return goal_pos_local[:, :2]  # (num_envs, 2)

def get_box_position(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.scene["box_1"].data.body_link_state_w[:, 0, :2]  # (B, 2)

def get_robot_position(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.scene["robot"].data.body_link_state_w[:, 0, :2]  # (B, 2)  

def get_trajectory_progress(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.trajectory_progress  # (B, 1)

def get_robot_velocity(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.scene["robot"].data.root_link_vel_w[:, :2]  # (B, 2)

def get_box_velocity(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.scene["box_1"].data.root_link_vel_w[:, :2]  # (B, 2)

def get_robot_move(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.body_link_state_w[:, 0, :2]  # (B, 2)
    robot_vel = env.scene["robot"].data.root_link_vel_w[:, :2]  # (B, 2)
    return torch.cat([robot_pos, robot_vel], dim=-1)  # (B, 4)

def get_box_move(
    env: ManagerBasedRLEnv,
    box_name: str = "box_1",
) -> torch.Tensor:
    box_pos = env.scene[box_name].data.body_link_state_w[:, 0, :2]  # (B, 2)
    box_vel = env.scene[box_name].data.root_link_vel_w[:, :2]  # (B, 2)
    return torch.cat([box_pos, box_vel], dim=-1)  # (B, 4)

def coordinate_transform(sys_pos: torch.Tensor, sys_quat: torch.Tensor, obj_pos: torch.Tensor) -> torch.Tensor:
    """
    Transform object position from world coordinates to system local coordinates.
    
    Args:
        sys_pos: System position in world frame (B, 3)
        sys_quat: System quaternion in world frame (B, 4) [w, x, y, z]
        obj_pos: Object position in world frame (B, 3)
        
    Returns:
        Object position in system local frame (B, 3)
    """
    pos_vec = obj_pos - sys_pos
    return quat_apply_inverse(sys_quat, pos_vec)