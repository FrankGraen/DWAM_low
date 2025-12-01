from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .actions import DifferentialDriveAction
import math

@configclass
class TurtlebotDiffDriveActionCfg(ActionTermCfg):
    """Configuration for the differential drive action term for Turtlebot."""

    # The action class type will be set in actions.py
    class_type: type[ActionTerm] = DifferentialDriveAction

    # scale: (linear, angular) velocity
    scale: tuple[float, float] = (1.0, math.radians(180))  # 1 m/s, 180 deg/s
    offset: tuple[float, float] = (0.0, 0.0)

    wheelbase_length: float = 0.160  # meters
    wheel_radius: float = 0.033      # meters

    # Joint names for left and right wheels
    drive_joint_names: list[str] = ["wheel_left_joint", "wheel_right_joint"]
