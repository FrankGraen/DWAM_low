from __future__ import annotations
from typing import TYPE_CHECKING

from isaacsim import SimulationApp
import carb
import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.assets.articulation import Articulation
from envs.robots.turtle_se import inverse_kinematics_tensor_from_root_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import TurtlebotDiffDriveActionCfg

class DifferentialDriveAction(ActionTerm):
    """
    Action term for the differential drive robot.
    """
    cfg: TurtlebotDiffDriveActionCfg

    _asset: Articulation

    _drive_joint_names: list[str]

    _drive_joint_ids: list[int]

    _scale: torch.Tensor

    _offset: torch.Tensor

    _raw_actions: torch.Tensor

    _processed_actions: torch.Tensor

    _joint_vel: torch.Tensor

    def __init__(self, cfg: TurtlebotDiffDriveActionCfg, env: 'ManagerBasedEnv'):
        super().__init__(cfg, env)
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)

        # save scale and offset for actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # (linear, angular)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        # Convert (linear, angular) to wheel velocities
        wheel_vels = inverse_kinematics_tensor_from_root_vel(
            torch.cat([
                self._processed_actions[:, 0:1],  # linear x
                torch.zeros(self.num_envs, 2, device=self.device),  # vy, vz
                torch.zeros(self.num_envs, 2, device=self.device),  # wx, wy
                self._processed_actions[:, 1:2]  # wz (yaw)
            ], dim=1),
            wheel_separation=self.cfg.wheelbase_length,
            wheel_radius=self.cfg.wheel_radius
        )
        self._joint_vel = wheel_vels
        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)

class DifferentialDriveActionNonVec:
    def __init__(self, cfg: TurtlebotDiffDriveActionCfg, robot: Articulation, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._asset = robot

        # Find drive joint IDs and names
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)

        # Save scale and offset for actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


    """
    Operations
    """

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        wheel_vels = inverse_kinematics_tensor_from_root_vel(
            torch.cat([
                self._processed_actions[:, 0:1],
                torch.zeros(self.num_envs, 2, device=self.device),
                torch.zeros(self.num_envs, 2, device=self.device),
                self._processed_actions[:, 1:2]
            ], dim=1),
            wheel_separation=self.cfg.wheelbase_length,
            wheel_radius=self.cfg.wheel_radius
        )
        self._joint_vel = wheel_vels
        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)

