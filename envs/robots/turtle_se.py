import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

TURTLEBOT3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(PROJECT_ROOT, "envs/robots/turtlebot3.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,#0.22 m/s
            max_angular_velocity=1000.0,#2.84 rad/s 
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*wheel.*": 0.0,
        },
        joint_vel={
            ".*wheel.*": 0.0,
        }
    ),
    actuators={
        "left_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_left_joint"],
            velocity_limit_sim=20.0,
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=1000.0,
        ),
        "right_wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_right_joint"],
            velocity_limit_sim=20.0,
            effort_limit_sim=20.0,
            stiffness=0.0,
            damping=1000.0,
        ),
    },
)
"""Configuration for turtlebot3_burger robot with active left/right wheels and passive joints."""

def inverse_kinematics_tensor_from_root_vel(
    root_vel: torch.Tensor,
    wheel_separation: float = 0.160,
    wheel_radius: float = 0.033
) -> torch.Tensor:
    """
    Extract left and right wheel angular velocities for differential drive from the root_vel tensor in Isaac Lab.

    Parameters:
    - root_vel: (N, 6) tensor, [vx, vy, vz, wx, wy, wz]
    - wheel_separation: Wheel separation distance (in meters)
    - wheel_radius: Wheel radius (in meters)

    Returns:
    - wheel_vels: (N, 2) tensor of left and right wheel angular velocities (in rad/s)
    """
    v = root_vel[:, 0]       # inear velocity in x direction
    omega = root_vel[:, 5]   # angular velocity around z axis (yaw)

    omega_left = (v - 0.5 * wheel_separation * omega) / wheel_radius
    omega_right = (v + 0.5 * wheel_separation * omega) / wheel_radius

    return torch.stack([omega_left, omega_right], dim=-1)  # [N, 2]
