"""
FOV-based observation functions for Isaac Lab environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple
import math
import torch
from .observations import *
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def is_in_fov_camera_relative_pos(
    obj_pos: torch.Tensor,
    width: float,
    height: float,
    focal_length: float,
    max_dist: float,
    is_camera_coordinate_system: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check if objects are in field of view given their positions in camera coordinate system.
    
    Args:
        obj_pos: Object positions in camera coordinate system (B, N, 3)
        width: Camera sensor width
        height: Camera sensor height
        focal_length: Camera focal length
        max_dist: Maximum detection distance
        is_camera_coordinate_system: If True, assume input is already in camera coordinates
        
    Returns:
        Tuple of (is_in_fov, is_in_dist) tensors
    """
    # Convert from robot coordinates to camera coordinates if needed
    if not is_camera_coordinate_system:
        # Robot coordinate system (x,y,z) -> Camera coordinate system (y,z,x)
        # x (forward) -> z (depth), y (right) -> x (image width), z (down) -> y (image height)
        obj_pos = obj_pos[..., [1, 2, 0]]
    
    # Check if object is in front of camera
    is_front = (obj_pos[..., 2] > 0).unsqueeze(-1)
    
    # Check distance constraint
    dist = torch.norm(obj_pos, dim=-1, keepdim=True)
    is_in_dist = (dist <= max_dist)
    
    # Use absolute values for projection calculation
    obj_pos_abs = torch.abs(obj_pos)
    
    # Project to image plane
    projection = obj_pos_abs * focal_length / obj_pos_abs[..., 2:3]
    
    # Check if projection is within sensor boundaries
    is_in_width = (projection[..., 0:1] <= width / 2.0)
    is_in_height = (projection[..., 1:2] <= height / 2.0)
    
    # Combine all conditions
    is_in_fov = (is_front & is_in_dist & is_in_width & is_in_height).squeeze(-1)
    
    return is_in_fov, is_in_dist.squeeze(-1)


def is_in_fov_angle(
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    obj_pos: torch.Tensor,
    h_fov: float,
    v_fov: float,
    focal_length: float,
    max_dist: float,
    is_deg: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check if objects are in field of view based on angular constraints.
    
    Args:
        cam_pos: Camera position in world frame (B, 3)
        cam_quat: Camera quaternion in world frame (B, 4) [w, x, y, z]
        obj_pos: Object positions in world frame (B, N, 3)
        h_fov: Horizontal field of view
        v_fov: Vertical field of view
        focal_length: Camera focal length
        max_dist: Maximum detection distance
        is_deg: Whether angles are in degrees
        
    Returns:
        Tuple of (is_in_fov, is_in_dist) tensors
    """
    if is_deg:
        h_fov = h_fov * math.pi / 180.0
        v_fov = v_fov * math.pi / 180.0
    
    # Convert to image plane dimensions
    height = 2.0 * math.tan(v_fov / 2.0) * focal_length
    width = 2.0 * math.tan(h_fov / 2.0) * focal_length
    
    # Transform object positions to camera coordinate system
    # Handle batch dimension properly
    batch_size = cam_pos.shape[0]
    num_objects = obj_pos.shape[1] if obj_pos.dim() > 2 else 1
    
    if obj_pos.dim() == 2:
        obj_pos = obj_pos.unsqueeze(1)  # Add object dimension
    
    # Expand camera pose for each object
    cam_pos_expanded = cam_pos.unsqueeze(1).expand(-1, num_objects, -1)  # (B, N, 3)
    cam_quat_expanded = cam_quat.unsqueeze(1).expand(-1, num_objects, -1)  # (B, N, 4)
    
    # Transform to camera coordinates
    obj_pos_camera = coordinate_transform(
        cam_pos_expanded.reshape(-1, 3),
        cam_quat_expanded.reshape(-1, 4),
        obj_pos.reshape(-1, 3)
    ).reshape(batch_size, num_objects, 3)
    
    # Check FOV
    return is_in_fov_camera_relative_pos(
        obj_pos_camera, width, height, focal_length, max_dist, is_camera_coordinate_system=False
    )

def get_fov_based_box_observation(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    box_names: List[str] = None,
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    mask_value: float = -10.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Get FOV-based observation of boxes from robot's perspective, returning distance and angle to each box.
    
    Args:
        env: The RL environment
        robot_name: Name of the robot asset
        box_names: List of box names to observe
        h_fov: Horizontal field of view
        v_fov: Vertical field of view
        focal_length: Camera focal length
        max_detection_distance: Maximum detection distance
        mask_value: Value to use for masked (invisible) observations
        is_deg: Whether FOV angles are in degrees
        
    Returns:
        FOV-based box observations tensor of shape (B, N * 2), where each box contributes [distance, angle]
    """
    if box_names is None:
        box_names = ["box_1"]
    
    robot = env.scene[robot_name]
    
    # Get robot pose
    robot_pos = robot.data.root_state_w[:, :3]  # (B, 3)
    robot_quat = robot.data.root_state_w[:, 3:7]  # (B, 4) [w, x, y, z]
    
    batch_size = robot_pos.shape[0]
    num_boxes = len(box_names)
    
    # Collect all box positions
    box_positions = []
    box_observations = []
    
    for box_name in box_names:
        # Get distance and angle to box
        distance = distance_to_box(env, box_name, robot_name)  # (B, 1)
        angle = angle_to_box(env, box_name, robot_name)  # (B, 1)
        
        # Get box position for FOV check
        box = env.scene[box_name]
        box_pos = box.data.body_link_state_w[:, 0, :3]  # (B, 3)
        box_positions.append(box_pos)
        
        # Combine distance and angle into observation for this box
        box_obs = torch.cat([distance, angle], dim=-1)  # (B, 2)
        box_observations.append(box_obs)
    
    if num_boxes > 1:
        box_positions = torch.stack(box_positions, dim=1)  # (B, N, 3)
        box_observations = torch.stack(box_observations, dim=1)  # (B, N, 2)
    else:
        box_positions = box_positions[0].unsqueeze(1)  # (B, 1, 3)
        box_observations = box_observations[0].unsqueeze(1)  # (B, 1, 2)
    
    # Check FOV visibility
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    # print(f"is_in_fov: {is_in_fov}")
    # print(f"is_in_dist: {is_in_dist}")
    
    # Apply masking to invisible objects
    masked_observations = box_observations.clone()
    invisible_mask = ~is_in_fov  # (B, N)
    invisible_mask_expanded = invisible_mask.unsqueeze(-1).expand_as(box_observations)  # (B, N, 2)
    masked_observations[invisible_mask_expanded] = mask_value
    # print(f"masked_observations: {masked_observations}")
    # print(f"reshape_masked_observations: {masked_observations.reshape(batch_size, -1)}")
    
    # Flatten for observation: (B, N * 2)
    return masked_observations.reshape(batch_size, -1)


def get_fov_visibility_flags(
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
    Get binary visibility flags for objects in FOV.
    
    Returns:
        Binary tensor indicating which objects are visible (B, N)
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
    is_in_fov, _ = is_in_fov_angle(
        robot_pos, robot_quat, box_positions,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    return is_in_fov.float()


def get_fov_based_goal_observation(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    mask_value: float = -10.0,
    is_deg: bool = True
) -> torch.Tensor:
    """
    Get FOV-based observation of goal position, returning distance, angle, and visibility flag.
    
    Args:
        env: The RL environment
        command_name: Name of the command to get goal position
        robot_name: Name of the robot asset
        h_fov: Horizontal field of view
        v_fov: Vertical field of view
        focal_length: Camera focal length
        max_detection_distance: Maximum detection distance
        mask_value: Value to use for masked (invisible) observations
        is_deg: Whether FOV angles are in degrees
        
    Returns:
        FOV-based goal observation tensor of shape (B, 3), containing [distance, angle, is_in_fov]
    """
    robot = env.scene[robot_name]
    robot_pos = robot.data.root_state_w[:, :3]  # (B, 3)
    robot_quat = robot.data.root_state_w[:, 3:7]  # (B, 4) [w, x, y, z]
    
    batch_size = robot_pos.shape[0]
    
    # Get distance and angle to goal using existing functions
    distance = distance_robot_to_goal(env)  # (B, 1)
    angle = angle_robot_to_goal(env)  # (B, 1)
    
    # Get goal position for FOV check
    goal_pos = env.goal_position  # (B, 3)
    
    # Expand goal position for FOV check
    goal_pos_expanded = goal_pos.unsqueeze(1)  # (B, 1, 3)
    
    # Check FOV visibility
    is_in_fov, is_in_dist = is_in_fov_angle(
        robot_pos, robot_quat, goal_pos_expanded,
        h_fov, v_fov, focal_length, max_detection_distance, is_deg
    )
    
    # Combine distance, angle, and visibility into observation
    goal_obs = torch.cat([distance, angle], dim=-1)  # (B, 2)
    goal_obs = goal_obs.unsqueeze(1)  # (B, 1, 2) to match box observation structure
    
    # Apply masking to invisible goal
    masked_observations = goal_obs.clone()
    invisible_mask = ~is_in_fov  # (B, 1)
    invisible_mask_expanded = invisible_mask.unsqueeze(-1).expand_as(goal_obs)  # (B, 1, 2)
    masked_observations[invisible_mask_expanded] = mask_value
    
    # Add visibility flag
    visibility_flag = is_in_fov.float()  # (B, 1)
    
    # Combine masked observations with visibility flag: [distance, angle, is_in_fov]
    final_obs = torch.cat([
        masked_observations.squeeze(1),  # (B, 2) - distance and angle
        visibility_flag  # (B, 1) - visibility flag
    ], dim=-1)  # (B, 3)
    return final_obs


# ============================================================================
# FOV Visualization Functions
# ============================================================================

def generate_fov_frustum_points(
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    width: float,
    height: float,
    focal_length: float,
    max_dist: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the corner points of an FOV frustum for visualization.
    
    Args:
        cam_pos: Camera position in world frame (B, 3)
        cam_quat: Camera quaternion in world frame (B, 4) [w, x, y, z]
        width: Camera sensor width
        height: Camera sensor height  
        focal_length: Camera focal length
        max_dist: Maximum detection distance
        
    Returns:
        Tuple of (frustum_corners, camera_centers) for drawing lines
        frustum_corners: (B, 4, 3) - four corner points of the frustum
        camera_centers: (B, 3) - camera center positions
    """
    batch_size = cam_pos.shape[0]
    device = cam_pos.device
    
    # Calculate scaling ratio to extend frustum to max distance
    ratio = max_dist / math.sqrt((width/2.0)**2 + (height/2.0)**2 + focal_length**2)
    
    # Define frustum corners in camera coordinate system
    # Camera coordinates: x=right, y=down, z=forward
    corners_camera = torch.tensor([
        [focal_length,  width/2.0,  height/2.0],   # top-right
        [focal_length, -width/2.0,  height/2.0],   # top-left  
        [focal_length, -width/2.0, -height/2.0],   # bottom-left
        [focal_length,  width/2.0, -height/2.0]    # bottom-right
    ], device=device, dtype=torch.float32) * ratio
    
    # Expand for batch processing
    corners_camera = corners_camera.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 4, 3)
    
    # Transform corners to world coordinates
    # Note: Need to handle coordinate system conversion from camera to robot coordinates
    # Camera: (x=forward, y=right, z=down) -> Robot: (x=forward, y=right, z=up)
    corners_robot = corners_camera[..., [0, 1, 2]]  # Keep same for now, adjust if needed
    
    # Rotate corners by camera quaternion and translate by camera position
    frustum_corners = quat_apply(cam_quat.unsqueeze(1), corners_robot) + cam_pos.unsqueeze(1)
    
    return frustum_corners, cam_pos


def generate_fov_frustum_lines(
    cam_pos: torch.Tensor, 
    cam_quat: torch.Tensor,
    width: float,
    height: float, 
    focal_length: float,
    max_dist: float,
    ground_z: float = 0.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate line segments for FOV frustum visualization with ground clipping.
    
    Args:
        cam_pos: Camera position in world frame (B, 3)
        cam_quat: Camera quaternion in world frame (B, 4)
        width: Camera sensor width
        height: Camera sensor height
        focal_length: Camera focal length
        max_dist: Maximum detection distance
        ground_z: Ground level Z coordinate for clipping
        
    Returns:
        Tuple of (start_points, end_points) lists for line drawing
    """
    frustum_corners, camera_centers = generate_fov_frustum_points(
        cam_pos, cam_quat, width, height, focal_length, max_dist
    )
    
    batch_size = cam_pos.shape[0]
    start_points = []
    end_points = []
    
    def project_line_to_ground(start: torch.Tensor, end: torch.Tensor, ground_z: float) -> torch.Tensor:
        """Project line segment endpoint to ground plane if it's below ground."""
        if end[2] >= ground_z:
            return end  # Point is above ground, no projection needed
        else:
            # Project point to ground plane (keep x, y coordinates, set z to ground_z)
            projected_end = end.clone()
            projected_end[2] = ground_z
            return projected_end
    
    for b in range(batch_size):
        corners = frustum_corners[b]  # (4, 3)
        center = camera_centers[b]   # (3,)
        
        # Rectangle edges (connect adjacent corners)
        for i in range(4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]
            
            projected_corner1 = project_line_to_ground(center, corner1, ground_z)
            projected_corner2 = project_line_to_ground(center, corner2, ground_z)
            
            start_points.append(projected_corner1)
            end_points.append(projected_corner2)
        
        # Lines from camera center to each corner (projected to ground if needed)
        for i in range(4):
            corner = corners[i]
            projected_corner = project_line_to_ground(center, corner, ground_z)
            
            start_points.append(center)
            end_points.append(projected_corner)
    
    return start_points, end_points


def visualize_fov_angle(
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor, 
    h_fov: float,
    v_fov: float,
    focal_length: float,
    max_dist: float,
    ground_z: float = 0.0,
    is_deg: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Visualize FOV using angular parameters with ground clipping.
    
    Args:
        cam_pos: Camera position in world frame (B, 3)
        cam_quat: Camera quaternion in world frame (B, 4)
        h_fov: Horizontal field of view
        v_fov: Vertical field of view  
        focal_length: Camera focal length
        max_dist: Maximum detection distance
        ground_z: Ground level Z coordinate for clipping
        is_deg: Whether angles are in degrees
        
    Returns:
        Tuple of (start_points, end_points) for line drawing
    """
    if is_deg:
        h_fov = h_fov * math.pi / 180.0
        v_fov = v_fov * math.pi / 180.0
    
    # Convert to image plane dimensions
    height = 2.0 * math.tan(v_fov / 2.0) * focal_length
    width = 2.0 * math.tan(h_fov / 2.0) * focal_length
    
    return generate_fov_frustum_lines(cam_pos, cam_quat, width, height, focal_length, max_dist, ground_z)


def get_robot_fov_visualization_data(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    h_fov: float = 90.0,
    v_fov: float = 60.0,
    focal_length: float = 1.0,
    max_detection_distance: float = 5.0,
    ground_z: float = 0.0,
    is_deg: bool = True,
    env_indices: torch.Tensor = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Get FOV visualization data for robot in Isaac Lab environment with ground clipping.
    
    Args:
        env: The RL environment
        robot_name: Name of the robot asset
        h_fov: Horizontal field of view
        v_fov: Vertical field of view
        focal_length: Camera focal length
        max_detection_distance: Maximum detection distance
        ground_z: Ground level Z coordinate for clipping (default: 0.0)
        is_deg: Whether FOV angles are in degrees
        env_indices: Environment indices to visualize (if None, visualize all)
        
    Returns:
        Tuple of (start_points, end_points) for drawing FOV frustums
    """
    robot = env.scene[robot_name]
    
    # Get robot pose
    robot_pos = robot.data.root_state_w[:, :3]  # (B, 3)
    robot_quat = robot.data.root_state_w[:, 3:7]  # (B, 4) [w, x, y, z]
    
    # Filter by environment indices if specified
    if env_indices is not None:
        robot_pos = robot_pos[env_indices]
        robot_quat = robot_quat[env_indices]

    return visualize_fov_angle(
        robot_pos, robot_quat, h_fov, v_fov, 
        focal_length, max_detection_distance, ground_z, is_deg
    )