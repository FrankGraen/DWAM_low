import os
import math
from dataclasses import MISSING


from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg

from envs.mdp.commands_cfg import BoxPoseCommandCfg
from envs.mdp.trajectory_cfg import TrajectoryConfig


from .robots import turtle_se
import envs.mdp as mdp


#cfg #TODO: move these to a config file
box_size = (0.15, 0.15, 0.15)  # Size of the boxes
box_mass = 0.5  # Mass of the boxes
curr_initial_radius = 1.0  # Initial bounding radius (used for curriculum)
bounding_radius = 2.2  # Final bounding radius (used for eval and as curriculum target)

# FOV Configuration
FOV_CONFIG = {
    "h_fov": 70.0,  # Horizontal field of view in degrees
    "v_fov": 60.0,  # Vertical field of view in degrees
    "focal_length": 1.0,  # Camera focal length
    "max_detection_distance": 5.0,  # Maximum detection distance
    "mask_value": -10.0,  # Value for masked observations
    "is_deg": True,  # FOV angles in degrees
    "visualize": True,  # Enable/disable FOV visualization
    "color": (255, 140, 0, 0.6),  # RGBA color for FOV visualization (semi-transparent orange)
    "ground_z": 0.0,  # Ground plane z-coordinate for visualization
}

# Trajectory Visualization Configuration
TRAJECTORY_VISUALIZATION_CONFIG = {
    "visualize": True,  # 是否启用轨迹可视化
    "color": (0.0, 1.0, 0.0, 0.8),  # RGBA：绿色，半透明
    "line_width": 3.0,
    "visualize_env_id": 0,  # 只可视化第一个环境
    "closed_loop": False,  # 是否闭合轨迹
    "show_closest_point": True,  # 是否显示最近点
}

import math

def calculate_camera_params(h_fov_deg, v_fov_deg, focal_length):
    """
    Calculate the horizontal and vertical aperture based on FOV and focal length.
    """
    h_fov_rad = math.radians(h_fov_deg)
    v_fov_rad = math.radians(v_fov_deg)

    # aperture = 2 * focal_length * tan(FOV / 2)
    horizontal_aperture = 2.0 * focal_length * math.tan(h_fov_rad / 2.0)
    vertical_aperture = 2.0 * focal_length * math.tan(v_fov_rad / 2.0)
    
    return horizontal_aperture, vertical_aperture

# Calculate camera parameters
h_aperture, v_aperture = calculate_camera_params(
    FOV_CONFIG["h_fov"], 
    FOV_CONFIG["v_fov"], 
    FOV_CONFIG["focal_length"]
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the robot scene."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(10000.0, 10000.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1e-4))
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # playground    
    playground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/playground",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=os.path.join(os.path.dirname(__file__), "playground.usd"), # radius 3m
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0.707, 0.0, 0.0)),
    )

    # Turtlebot3 Robot
    robot: ArticulationCfg = turtle_se.TURTLEBOT3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_footprint/base_link/front_cam",
    #     update_period=0.1,
    #     width=105,
    #     height=90,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=FOV_CONFIG["focal_length"],  
    #         horizontal_aperture=h_aperture,           
    #         vertical_aperture=v_aperture,             
    #         clipping_range=(0.01, FOV_CONFIG["max_detection_distance"]),
    #         visible=True,             
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.035, 0.0, 0.18), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(link|Link)",
    )

    # box
    box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box1",
        spawn=sim_utils.CuboidCfg(
            size=box_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=box_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # blue
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )
    # box_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Box2",
    #     spawn=sim_utils.CuboidCfg(
    #         size=box_size,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=box_mass),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # red
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #             restitution=0.8,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    # )
    # box_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Box3",
    #     spawn=sim_utils.CuboidCfg(
    #         size=box_size,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=box_mass),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # green
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #             restitution=0.8,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.05)),
    # )

    # trajectory

@configclass
class ActionCfg:
    """Action configuration for the task."""
    
    actions: ActionTerm = mdp.TurtlebotDiffDriveActionCfg(
        asset_name="robot",
    )

@configclass
class ObservationCfg:
    """Observation configuration for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Keep last action
        actions = ObsTerm(func=mdp.last_action)

        # # Original distance and angle observations (for comparison)
        # distance_robot_box = ObsTerm(
        #     func=mdp.distance_to_box,
        #     params={"box_name": "box_1", "asset_name": "robot"},
        #     scale=0.1
        # )

        # angle_robot_box = ObsTerm(
        #     func=mdp.angle_to_box,
        #     params={"box_name": "box_1", "asset_name": "robot"},
        #     scale=1.0 / math.pi  # Normalize to [-1, 1]
        # )

        # FOV-based box observation
        fov_box_observation = ObsTerm(
            func=mdp.get_fov_based_box_observation,
            params={
                "robot_name": "robot",
                "box_names": ["box_1"],
                "h_fov": FOV_CONFIG["h_fov"],
                "v_fov": FOV_CONFIG["v_fov"],
                "focal_length": FOV_CONFIG["focal_length"],
                "max_detection_distance": FOV_CONFIG["max_detection_distance"],
                "mask_value": FOV_CONFIG["mask_value"],
                "is_deg": FOV_CONFIG["is_deg"]
            },
            scale=0.1  # Scale the relative positions and velocities
        )

        # FOV visibility flags for boxes
        # fov_visibility_flags = ObsTerm(
        #     func=mdp.get_fov_visibility_flags,
        #     params={
        #         "robot_name": "robot",
        #         "box_names": ["box_1"],
        #         "h_fov": FOV_CONFIG["h_fov"],
        #         "v_fov": FOV_CONFIG["v_fov"],
        #         "focal_length": FOV_CONFIG["focal_length"],
        #         "max_detection_distance": FOV_CONFIG["max_detection_distance"],
        #         "is_deg": FOV_CONFIG["is_deg"]
        #     },
        #     scale=1.0
        # )

        # FOV-based goal observation
        fov_goal_observation = ObsTerm(
            func=mdp.get_fov_based_goal_observation,
            params={
                "robot_name": "robot",
                "h_fov": FOV_CONFIG["h_fov"],
                "v_fov": FOV_CONFIG["v_fov"],
                "focal_length": FOV_CONFIG["focal_length"],
                "max_detection_distance": FOV_CONFIG["max_detection_distance"],
                "mask_value": FOV_CONFIG["mask_value"],
                "is_deg": FOV_CONFIG["is_deg"]
            },
            scale=0.1
        )

        # Original goal observations (for comparison)
        # distance_robot_goal = ObsTerm(
        #     func=mdp.distance_robot_to_goal,
        #     params={"command_name": "box_target"},
        #     scale=0.1
        # )

        # angle_robot_goal = ObsTerm(
        #     func=mdp.angle_robot_to_goal,
        #     params={"command_name": "box_target"},
        #     scale=1.0 / math.pi  # Normalize to [-1, 1]
        # )
        
        # robot_move_observation = ObsTerm(
        #     func=mdp.get_robot_move,
        #     params={},
        #     scale=1.0,
        # )
        
        # box_move_observation = ObsTerm(
        #     func=mdp.get_box_move,
        #     params={"box_name": "box_1"},
        #     scale=1.0,
        # )
        # goal_observation = ObsTerm(
        #     func=mdp.get_goal_observation,
        #     params={},
        #     scale=0.1,
        # )
        trajectory_observation = ObsTerm(
            func=mdp.get_trajectory_history_and_future,
            params={
                "past_steps": 0,
                "future_steps": 2,
                "flatten": True,
            },
            scale=0.1,  # Scale the trajectory points
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    # @configclass
    # class TrajectoryCfg(ObsGroup):
    #     trajectory_observation = ObsTerm(
    #         func=mdp.get_trajectory_history_and_future,
    #         params={
    #             "past_steps": 10,
    #             "future_steps": 30,
    #         },
    #         scale=1.0,  # Scale the trajectory points
    #     )
        
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

    # @configclass
    # class StudentObsCfg(ObsGroup):

    #     # RGB-D observation for student network (not flattened)
    #     camera_rgbd = ObsTerm(
    #         func=mdp.get_camera_rgbd,
    #         params={
    #             "sensor_cfg": SceneEntityCfg("camera"),
    #             "normalize": True
    #         },
    #         scale=1.0
    #     )
        
    #     prev_action = ObsTerm(func=mdp.last_action)

    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # trajectory: TrajectoryCfg = TrajectoryCfg()

@configclass
class RewardsCfg:
    # Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # robot to box distance reward
    # robot_distance_to_box = RewTerm(
    #     func=mdp.distance_to_box_reward,
    #     weight=0.004,
    #     params={"box_name": "box_1", "asset_name": "robot"},
    # )

    # robot to box angle reward
    robot_angle_to_box = RewTerm(
        func=mdp.angle_to_box_reward,
        weight=0.003,
        params={"box_name": "box_1", "asset_name": "robot", "distance_threshold": 0.5},
    )

    # decrease reward
    distance_decrease = RewTerm(
        func=mdp.distance_decrease_reward,
        weight=0.01,
        params={},
    )

    # robot reached box one-time reward
    reached_box = RewTerm(
        func=mdp.reached_box_reward,
        weight=10.0,
        params={
            "box_name": "box_1",
            "distance_threshold": 0.20,  # Distance threshold to consider the box reached
        },
    )

    # Box reached target one-time reward
    # box_reached_target = RewTerm(
    #     func=mdp.box_reached_target_reward,
    #     weight=50.0,
    #     params={
    #         "command_name": "box_target",
    #         "box_name": "box_1",
    #         "threshold_pos": 0.1,
    #         "threshold_rot": 0.1,
    #         "speed_threshold": 0.05,
    #     },
    # )

    # Encourage slowing down near the goal to avoid overshoot
    # slowdown_near_target = RewTerm(
    #     func=mdp.slowdown_near_target_reward,
    #     weight=0.05,
    #     params={
    #         "box_name": "box_1",
    #         "command_name": "box_target",
    #         "slowdown_distance": 0.35,
    #         "speed_threshold": 0.2,
    #     },
    # )

    # Box to goal distance reward
    # box_distance_to_target = RewTerm(
    #     func=mdp.distance_to_target_reward,
    #     weight=0.04,
    #     params={"command_name": "box_target"},
    # )

    # Speed smoothness reward
    oscillation = RewTerm(
        func=mdp.oscillation_penalty,
        weight=-0.0002, 
        params={},
    )

    # Out of bounds penalty
    out_of_bounds = RewTerm(
        func=mdp.out_of_bounds_penalty,
        weight=-0.01,
        params={
            "radius": bounding_radius,
        },
    )

    # NEW: FOV-based exploration reward
    fov_exploration = RewTerm(
        func=mdp.fov_exploration_reward,
        weight=0.001,
        params={
            "robot_name": "robot",
            "box_names": ["box_1"],
            "h_fov": FOV_CONFIG["h_fov"],
            "v_fov": FOV_CONFIG["v_fov"],
            "focal_length": FOV_CONFIG["focal_length"],
            "max_detection_distance": FOV_CONFIG["max_detection_distance"],
            "is_deg": FOV_CONFIG["is_deg"]
        },
    )
    
    # Trajectory following reward
    # trajectory_following = RewTerm(
    #     func=mdp.trajectory_following_reward,
    #     weight=1.0,
    #     params={
    #         "distance_threshold": 0.2,
    #         "distance_weight":  1.0,
    #         "progress_weight": 2.0,
    #         "velocity_alignment_weight": 0.8,
    #         "off_track_penalty": -1.5,
    #         "backward_penalty": -0.8,
    #         "distance_jump_penalty": -1.0,
    #     },
    # )
    
    trajectory_distance_reward = RewTerm(
        func=mdp.trajectory_distance_reward,
        weight=0.05,
        params={
            "distance_threshold": 0.2,
        },
    )
    
    # trajectory_progress_reward = RewTerm(
    #     func=mdp.trajectory_progress_reward,
    #     weight=0.015,
    #     params={},
    # )
    
    # trajectory_following_reward_milestone = RewTerm(
    #     func=mdp.trajectory_following_reward_milestone,
    #     weight=8.0,
    #     params={
    #         "milestone": 0.5,
    #         "milestone_reward": 1.0,
    #         "distance_threshold": 0.2,
    #     },
    # )
    
    # trajectory_following_reward_milestone2 = RewTerm(
    #     func=mdp.trajectory_following_reward_milestone,
    #     weight=10.0,
    #     params={
    #         "milestone": 0.6,
    #         "milestone_reward": 1.0,
    #         "distance_threshold": 0.2,
    #     },
    # )
    
    # trajectory_following_reward_milestone3 = RewTerm(
    #     func=mdp.trajectory_following_reward_milestone,
    #     weight=10.0,
    #     params={
    #         "milestone": 0.9,
    #         "milestone_reward": 1.0,
    #         "distance_threshold": 0.2,
    #     },
    # )

    # trajectory_trajectory_velocity_alignment_reward = RewTerm(
    #     func=mdp.trajectory_velocity_alignment_reward,
    #     weight=0.01,
    #     params={},
    # )
    
    # trajectory_off_track_penalty = RewTerm(
    #     func=mdp.trajectory_off_track_penalty,
    #     weight=-0.01,
    #     params={},
    # )
    
    # trajectory_backward_penalty = RewTerm(
    #     func=mdp.trajectory_backward_penalty,
    #     weight=-0.006,
    #     params={},
    # )
    
    # trajectory_jump_penalty = RewTerm(
    #     func=mdp.trajectory_jump_penalty,
    #     weight=-0.002,
    #     params={
    #         "jump_threshold": 0.05,    
    #     },
    # )
    
    # Trajectory progress finish reward
    trajectory_progress_finish = RewTerm(
        func=mdp.trajectory_progress_finish_reward,
        weight=25.0,
        params={
            "progress_threshold": 0.95,
            "distance_threshold": 0.2,
            "speed_threshold": 0.1,
            "reward_amount": 5.0,
        },
    )
    
    # Slow down near finish reward
    slow_down_near_finish = RewTerm(
        func=mdp.slow_down_near_finish_reward,
        weight=0.04,
        params={
            "box_name": "box_1",
            "progress_threshold": 0.8,
            "speed_threshold": 0.2,
        },
    )

#     collision = RewTerm(
#         func=mdp.collision_penalty,
#         weight=-2.0,
#         params={"sensor_cfg": SceneEntityCfg(
#             "contact_sensor"), "threshold": 1.0},
#     )

@configclass
class TerminationsCfg:
    """Termination conditions for the task."""
    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    
    is_success = DoneTerm(
        func=mdp.is_success,
        params={
            "command_name": "box_follow_trajectory_finished",
            "box_name": "box_1",
            "progress_threshold": 0.99,
            "distance_threshold": 0.2,
            "speed_threshold": 0.01,
        },
    )
    
#     collision = DoneTerm(
#         func=mdp.collision_with_obstacles,
#         params={
#             "sensor_cfg": SceneEntityCfg("contact_sensor"), 
#             "threshold": 1.0
#         },
#     )
#     jackknifing_over = DoneTerm(
#         func=mdp.jackknifing_over,
#         params={
#             "command_name": "target", 
#             "navigation_mode": HEADING_MODE,                 
#         },
#     )
#     off_board = DoneTerm(
#         func=mdp.off_board,
#         params={"threshold": 0.0},
#     )  


@configclass
class CommandsCfg:
    box_target = BoxPoseCommandCfg(
    box_name="box_1",
    robot_name="robot",
    avoid_pattern="box_.*",
    sampling_radius=bounding_radius,
    resampling_time_range=(1500.0, 1500.0), # set to a large value to avoid resampling before the episode ends
    debug_vis=True,
    symmetric_rotation=True
)
    
@configclass
class EventCfg:
    """Randomization configuration for the task."""
    reset_state = EventTerm(
        func=mdp.reset_root_state,
        mode="reset",
        params={
            "robot_name": "robot",
            "box_name": "box_1",
            "sampling_radius": bounding_radius,
            "z_offset": 0.05,
        },
    )
    
    init_trajectory = EventTerm(
        func=mdp.init_trajectory,
        mode="reset",
        params={},
    )

@configclass
class CurriculumCfg:
    """Curriculum learning configuration for the task."""
    radius_curriculum = CurrTerm(
        func=mdp.radius_curriculum,
        params={
            "kwargs": {   # parameters for curriculum learning
                "initial_radius": curr_initial_radius,
                "final_radius": bounding_radius,  # Use bounding_radius as final target
                "success_rate_threshold": 0.7,
                "min_episodes_per_stage": 100,
                "adaptive_increment": True,
                "window_min": 100,
                "window_k": 2,
            }
        },
    )
    
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the robot environment."""
    
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1, env_spacing=6.5)


     # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**21,   # 2**13,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )

    actions: ActionCfg = ActionCfg()
    observations: ObservationCfg = ObservationCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # FOV Visualization Configuration
    fov_config = FOV_CONFIG
    # trajectory_cfg: TrajectoryConfig = TrajectoryConfig()
    trajectory_visualization = TRAJECTORY_VISUALIZATION_CONFIG

    def __post_init__(self):
        super().__post_init__()
        
        self.sim.dt = 1 / 60.0
        self.decimation = 6
        self.episode_length_s = 100
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
        # if self.scene.camera is not None:
        #     self.scene.camera.update_period = self.sim.dt * self.decimation