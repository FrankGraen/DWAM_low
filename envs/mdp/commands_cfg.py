from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import CommandTermCfg
from typing import Tuple
from .randomizations import BoxPoseCommand
from dataclasses import MISSING

@configclass
class BoxPoseCommandCfg(CommandTermCfg):
    """Configuration for box target pose command."""

    class_type: type = BoxPoseCommand

    box_name: str = MISSING
    """Box asset name."""

    robot_name: str = MISSING
    """Robot asset name."""

    avoid_pattern: str = MISSING
    """Regex pattern for other boxes to avoid sampling collisions."""

    sampling_radius: float = MISSING
    """Radius of the circular area for target position sampling."""

    debug_vis: bool = MISSING
    """Whether to show debug marker arrows for targets."""

    symmetric_rotation: bool = MISSING
    """Whether the box has symmetric orientation (e.g., cube)."""
