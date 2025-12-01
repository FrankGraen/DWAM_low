from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse, wrap_to_pi, yaw_quat, quat_conjugate, quat_mul
import isaaclab.sim as sim_utils

import re
import torch
import math
import random
from typing import Sequence

SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

# -------------------- Command Term for Box Pose --------------------
class BoxPoseCommand(CommandTerm):
    """Command generator for box target pose.
    The position commands are sampled within a specified range, avoiding collisions with other boxes.
    The orientation is sampled based on the box's symmetry.
    """
    def __init__(self, cfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        # Get the box and robot objects from the environment
        self.box: RigidObject = env.scene[cfg.box_name]
        self.robot: Articulation = env.scene[cfg.robot_name]

        # Get other boxes to avoid collisions FIXME: NOT WORKING
        # self.other_boxes = [
        #     obj for name, obj in env.scene.object_dict.items()
        #     if re.fullmatch(cfg.avoid_pattern, name) and name != cfg.box_name
        # ]

        # Create buffers to store the command (pose) and metric
        # -- commands: desired goal pose (pos + quat)
        self.goal_pose_w = torch.zeros(self.num_envs, 7, device=self.device)  # [x, y, z, qw, qx, qy, qz]
        self.goal_pose_b = torch.zeros_like(self.goal_pose_w)

        # -- metrics: error in heading and position
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_rot"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BoxPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling range: {self.cfg.target_position_range}\n"
        msg += f"\tSymmetric rotation: {self.cfg.symmetric_rotation}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        Return the command tensor containing the target pose for the box.
        (num_envs, 7) -> [pos(x,y,z), quat(x,y,z,w)]
        """
        return self.goal_pose_b

    def _resample_command(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        pos = self._sample_valid_position(env_ids)
        rot = self._sample_orientation(env_ids)
        
        # Combine position and rotation into a single pose tensor
        self.goal_pose_w[env_ids] = torch.cat([pos, rot], dim=1)

    def _update_command(self):
        """Re-target the position and orientation of the box to its body frame.
        The final command is the distance and rotation of the box relative to its target pose.
        """
        # Extract position and rotation from goal_pose_w and box's current pose
        goal_pos_w = self.goal_pose_w[:, :3]  # (num_envs, 3)
        goal_rot_w = self.goal_pose_w[:, 3:]  # (num_envs, 4)
        
        # Extract position and quaternion from body_link_state_w
        # body_link_state_w shape: (num_envs, 1, 13) -> [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        box_state = self.box.data.body_link_state_w.squeeze(1)  # (num_envs, 13)
        box_pos_w = box_state[:, :3]  # (num_envs, 3) - position
        box_rot_w = box_state[:, 3:7]  # (num_envs, 4) - quaternion

        # Compute relative position in world frame
        target_vec = goal_pos_w - box_pos_w  # (num_envs, 3)

        # Transform position to box's body frame using inverse rotation
        box_yaw_quat = yaw_quat(box_rot_w)  # (num_envs, 4)
        pos_b = quat_apply_inverse(box_yaw_quat, target_vec)  # (num_envs, 3)

        # Transform rotation to box's body frame
        box_rot_w_inv = quat_conjugate(box_rot_w)  # Inverse of box's rotation
        rot_b = quat_mul(box_rot_w_inv, goal_rot_w)  # Relative rotation in body frame

        # Combine position and rotation into goal_pose_b
        self.goal_pose_b[:] = torch.cat([pos_b, rot_b], dim=1)  # (num_envs, 7)


    def _update_metrics(self):
        # Extract position and quaternion from body_link_state_w
        # body_link_state_w shape: (num_envs, 1, 13) -> [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        box_state = self.box.data.body_link_state_w.squeeze(1)  # (num_envs, 13)
        box_pos = box_state[:, :3]  # (num_envs, 3) - position
        box_rot = box_state[:, 3:7]  # (num_envs, 4) - quaternion
        
        # Compute position error (Euclidean distance)
        pos_error = torch.norm(self.goal_pose_w[:, :3] - box_pos, dim=1)  # (num_envs,)
        self.metrics["error_pos"][:] = pos_error

        # Compute rotation error (angle between quaternions)
        goal_rot = self.goal_pose_w[:, 3:]  # (num_envs, 4)
        dot_product = torch.sum(goal_rot * box_rot, dim=1)  # (num_envs,)
        dot_product = torch.abs(dot_product)  # Handle q and -q equivalence
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Avoid numerical errors
        rot_error = 2.0 * torch.acos(dot_product)  # (num_envs,)
        self.metrics["error_rot"][:] = rot_error

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the visibility of debug visualization markers for the box's target pose."""
        if debug_vis:
            if not hasattr(self, "arrow_goal_visualizer"):
                arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                arrow_cfg.prim_path = "/Visuals/Command/heading_goal"
                arrow_cfg.markers["arrow"].scale = (0.1, 0.1, 0.4)
                self.arrow_goal_visualizer = VisualizationMarkers(arrow_cfg)
            if not hasattr(self, "sphere_goal_visualizer"):
                sphere_cfg = SPHERE_MARKER_CFG.copy()
                sphere_cfg.prim_path = "/Visuals/Command/position_goal"
                sphere_cfg.markers["sphere"].radius = 0.05 # TODO: move to config file
                self.sphere_goal_visualizer = VisualizationMarkers(sphere_cfg)

            # Set visibility to true
            self.arrow_goal_visualizer.set_visibility(True)
            self.sphere_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "arrow_goal_visualizer"):
                self.arrow_goal_visualizer.set_visibility(False)
            if hasattr(self, "sphere_goal_visualizer"):
                self.sphere_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization markers for the box's target pose."""
        # Extract position and quaternion from goal_pose_w
        goal_pos_w = self.goal_pose_w[:, :3]  # (num_envs, 3)
        goal_rot_w = self.goal_pose_w[:, 3:]  # (num_envs, 4)

        # Update the sphere marker to show the target position
        self.sphere_goal_visualizer.visualize(goal_pos_w)

        # Update the arrow marker to show the target orientation
        # Offset the arrow position slightly above the goal position for visibility
        position_arrow_w = goal_pos_w + torch.tensor([0.0, 0.0, 0.25], device=self.device)
        self.arrow_goal_visualizer.visualize(position_arrow_w, goal_rot_w)

    def _sample_valid_position(self, env_ids):
        """
        Sample random positions for specified environments relative to each environment's origin.
        Positions are sampled within a circular area with radius R, with z = 0.

        Args:
            env_ids: Tensor or list of environment indices to sample positions for.
        
        Returns:
            position: Tensor of shape (len(env_ids), 3) containing absolute x, y, z coordinates.
        """
        # Get radius from configuration
        radius = self.cfg.sampling_radius

        # Get the environment origins for the specified env_ids
        env_origins = self.env.scene.env_origins[env_ids]

        # Initialize relative position buffer
        relative_position = torch.zeros(len(env_ids), 3, device=self.device)

        # Sample positions uniformly within a circle of radius R
        # Use polar coordinates: r ~ sqrt(uniform), θ ~ uniform
        r = torch.sqrt(torch.rand(len(env_ids), device=self.device)) * radius  # Uniform distribution in circle
        theta = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi  # Random angle [0, 2π]

        # Convert polar to cartesian coordinates
        relative_position[:, 0] = r * torch.cos(theta)  # x coordinate
        relative_position[:, 1] = r * torch.sin(theta)  # y coordinate
        relative_position[:, 2] = 0.0  # z fixed at 0 since no terrain

        # Compute absolute positions by adding environment origins
        absolute_position = env_origins + relative_position

        return absolute_position

    def _sample_orientation(self, env_ids):
        """
        Sample random orientations (yaw angles) for specified environments.
        Orientations are represented as yaw angles in [-pi, pi] and converted to quaternions.

        Args:
            env_ids: Tensor or list of environment indices to sample orientations for.
        
        Returns:
            quaternion: Tensor of shape (len(env_ids), 4) containing [qw, qx, qy, qz].
        """
        # Sample random yaw angles in [-pi, pi]
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)

        # Convert yaw to quaternions (qw, qx, qy, qz)
        quaternion = torch.zeros(len(env_ids), 4, device=self.device)
        quaternion[:, 0] = torch.cos(yaw / 2.0)  # qw = cos(yaw/2)
        quaternion[:, 3] = torch.sin(yaw / 2.0)  # qz = sin(yaw/2), assuming rotation around z-axis

        return quaternion
# -------------------------------------------------------

def reset_root_state(
    env: ManagerBasedEnv, env_ids: torch.Tensor, robot_name: str, box_name: str, 
    sampling_radius: float = 1.0, z_offset: float = 0.05
):
    """
    Reset the root state (position and rotation) of both the robot and the box relative to each environment's origin.
    Positions are sampled within a circular area.

    Args:
        env: The simulation environment containing the scene and assets.
        env_ids: Tensor of indices identifying environments to reset.
        robot_name: Name of the robot asset.
        box_name: Name of the box asset.
        sampling_radius: Radius of the circular sampling area (default: 1.0).
        z_offset: Vertical offset to avoid spawning inside terrain (default: 0.05).
    """
    # Get the robot and box assets
    robot: RigidObject = env.scene[robot_name]
    box: RigidObject = env.scene[box_name]

    # Get the environment origins
    env_origins = env.scene.env_origins[env_ids]

    # --- Box ---
    # Sample position within a circle of radius sampling_radius
    r_box = torch.sqrt(torch.rand(len(env_ids), device=env.device)) * sampling_radius
    theta_box = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi

    box_rel_pos = torch.zeros(len(env_ids), 3, device=env.device)
    box_rel_pos[:, 0] = r_box * torch.cos(theta_box)  # x coordinate
    box_rel_pos[:, 1] = r_box * torch.sin(theta_box)  # y coordinate
    box_rel_pos[:, 2] = z_offset
    box_pos = env_origins + box_rel_pos

    # Random orientation (yaw only)
    box_angle = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi  # Random angle [0, 2π]
    box_quat = torch.zeros(len(env_ids), 4, device=env.device)
    box_quat[:, 0] = torch.cos(box_angle / 2)  # w component
    box_quat[:, 3] = torch.sin(box_angle / 2)  # z component (yaw)
    box_orient = box_quat

    box_pose = torch.cat([box_pos, box_orient], dim=-1)
    box.write_root_link_pose_to_sim(box_pose, env_ids=env_ids)

    # --- Robot ---
    # Sample robot position within a circle of radius sampling_radius
    r_robot = torch.sqrt(torch.rand(len(env_ids), device=env.device)) * sampling_radius
    theta_robot = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi

    robot_rel_pos = torch.zeros(len(env_ids), 3, device=env.device)
    robot_rel_pos[:, 0] = r_robot * torch.cos(theta_robot)  # x coordinate
    robot_rel_pos[:, 1] = r_robot * torch.sin(theta_robot)  # y coordinate
    robot_rel_pos[:, 2] = z_offset
    robot_pos = env_origins + robot_rel_pos

    # Randomize rotation (yaw only)
    robot_angle = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi  # Random angle [0, 2π]
    robot_quat = torch.zeros(len(env_ids), 4, device=env.device)
    robot_quat[:, 0] = torch.cos(robot_angle / 2)  # w component
    robot_quat[:, 3] = torch.sin(robot_angle / 2)  # z component (yaw)
    robot_orient = robot_quat

    robot_pose = torch.cat([robot_pos, robot_orient], dim=-1)
    robot.write_root_link_pose_to_sim(robot_pose, env_ids=env_ids)