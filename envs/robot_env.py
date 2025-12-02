import torch
import os
from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from datetime import datetime
import numpy as np

from .robot_env_cfg import RobotEnvCfg
from envs.mdp import get_distance  # Import get_distance for reset
from envs.mdp.trajectory_cfg import TrajectoryGenerator # Import TrajectoryGenerator

# observations, rewards, terminations, truncations, and extras (info dictionary)
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor,
                         torch.Tensor, torch.Tensor, dict]

class RobotEnv(ManagerBasedRLEnv):
    """ Robot environment for box manipulation tasks."""

    def __init__(self, cfg: RobotEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # env_ids = torch.arange(self.num_envs, device=self.device)

        # Initialize environment-specific variables
        self.global_step_counter = 0
        self.last_angle_bottom = torch.zeros(self.num_envs, device=self.device)
        # Buffers for distance decrease reward
        self._prev_robot_box_dist = torch.zeros(self.num_envs, device=self.device)  # (num_envs,)
        self._prev_box_goal_dist = torch.zeros(self.num_envs, device=self.device)  # (num_envs,)

    # ------------------------------ FOV Visualization setup -------------------------------
        self.debug_draw = None
        # Get visualization setting from config
        fov_config = getattr(cfg, 'fov_config', {})
        self.fov_visualization_enabled = fov_config.get('visualize', False)
        
        trajectory_vis_config = getattr(cfg, 'trajectory_visualization', {})
        self.trajectory_visualization_enabled = trajectory_vis_config.get('visualize', False)

        
        # Initialize debug draw interface
        if self.fov_visualization_enabled or self.trajectory_visualization_enabled:
            self._init_debug_draw()
    
    # ------------------------------ Trajectory recorder setup -------------------------------
        self.trajectory_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Progress index for each env
        self.closest_point_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.trajectory_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.trajectory_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_closest_points = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._prev_trajectory_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    
    # ------------------------------- Print Rewards setup -------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reward_log_dir : str = os.path.join("logs", "rewards", timestamp)
        if not os.path.exists(self.reward_log_dir):
            os.makedirs(self.reward_log_dir)
            print(f"Created reward log directory at: {self.reward_log_dir}")
        else:
            print(f"Reward log directory already exists at: {self.reward_log_dir}")

    def print_rewards(self):
        """打印奖励详情"""
        print("\n=== Reward Details (Env 0) ===")
        print(f"Total: {self.reward_buf[0].item():.4f}\n")
        
        for idx, name in enumerate(self.reward_manager._term_names):
            # 获取当前步的奖励值
            value = self.reward_manager._step_reward[0, idx].item()
            weight = self.reward_manager.get_term_cfg(name).weight
            
            print(f"  {name:40s}: {value:8.4f} (w={weight:6.3f})")
            
            reward_file = os.path.join(self.reward_log_dir, f"{name}.txt")
            if not os.path.exists(reward_file):
                with open(reward_file, "w") as f:
                    f.write(f"Reward: {name} (weight={weight})\n")
            with open(reward_file, "a") as f:
                f.write(f"{value}\n")
        print("================================\n")
    
    def print_observations(self):
        """打印观测详情（专为 Isaac Lab ObservationManager 优化）"""
        print("\n=== Observation Details (Env 0) ===")

        # 直接取 policy 组的完整观测（已拼接好）
        full_obs = self.observation_manager._obs_buffer["policy"][0].cpu().numpy()
        names = self.observation_manager.active_terms["policy"]
        dims = self.observation_manager.group_obs_term_dim["policy"]

        ptr = 0
        for name, dim in zip(names, dims):
            value = full_obs[ptr:ptr + dim[0]]  # dim 是 tuple 如 (4,)
            print(f"  {name:30s} {str(value.shape):<8} {np.array2string(value, precision=4, suppress_small=True)}")
            ptr += dim[0]

        print("=====================================\n")

    def _init_debug_draw(self):
        """Initialize the debug draw interface for FOV visualization."""
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
            print("Debug draw interface initialized successfully")
        except ImportError:
            print("Warning: Could not import debug_draw. FOV visualization will be disabled.")
            self.fov_visualization_enabled = False
        except Exception as e:
            print(f"Warning: Failed to initialize debug draw: {e}. FOV visualization will be disabled.")
            self.fov_visualization_enabled = False

    def _update_fov_visualization(self):
        """Update FOV visualization for all robots."""
        if not self.fov_visualization_enabled or self.debug_draw is None:
            return
            
        try:
            # Import the FOV visualization function
            from envs.mdp.fov_observations import get_robot_fov_visualization_data
            
            # Get FOV configuration from environment config
            fov_config = getattr(self.cfg, 'fov_config', {
                "h_fov": 70.0,
                "v_fov": 60.0,
                "focal_length": 1.0,
                "max_detection_distance": 5.0,
                "is_deg": True
            })
            
            # Clear previous FOV visualization
            self.debug_draw.clear_lines()
            
            # Get FOV visualization data
            start_points, end_points = get_robot_fov_visualization_data(
                env=self,
                robot_name="robot",
                h_fov=fov_config["h_fov"],
                v_fov=fov_config["v_fov"],
                focal_length=fov_config["focal_length"],
                max_detection_distance=fov_config["max_detection_distance"],
                is_deg=fov_config["is_deg"],
                env_indices=None,  # Visualize all environments
                ground_z=fov_config["ground_z"],
            )
            
            if start_points and end_points:
                # Convert tensor points to tuples for Isaac Sim
                start_points_list = []
                end_points_list = []
                
                for start_p, end_p in zip(start_points, end_points):
                    if torch.is_tensor(start_p):
                        start_points_list.append(tuple(start_p.cpu().numpy().astype(float)))
                    else:
                        start_points_list.append(tuple(float(x) for x in start_p))
                        
                    if torch.is_tensor(end_p):
                        end_points_list.append(tuple(end_p.cpu().numpy().astype(float)))
                    else:
                        end_points_list.append(tuple(float(x) for x in end_p))
                
                # Set FOV visualization style
                num_lines = len(start_points_list)
                colors = [fov_config["color"]] * num_lines  # Semi-transparent orange
                widths = [2.0] * num_lines  # Line width
                
                # Draw FOV frustums
                self.debug_draw.draw_lines(start_points_list, end_points_list, colors, widths)
                
        except Exception as e:
            print(f"Warning: Failed to update FOV visualization: {e}")
            # Disable visualization on repeated failures
            self.fov_visualization_enabled = False
    # ------------------------------ End of FOV Visualization setup -------------------------------
    
    # ------------------------------ Trajectory Visualization setup -------------------------------
    def _update_trajectory_visualization(self):
        """Update trajectory visualization for all environments."""
        if not self.trajectory_visualization_enabled or self.debug_draw is None:
            return
        
        if self.current_trajectories is None:
            return
        
        try:
            # 获取配置
            traj_config = getattr(self.cfg, 'trajectory_visualization', {
                "color": (0.0, 1.0, 0.0, 0.8),  # 绿色
                "line_width": 3.0,
                "visualize_env_id": 0,  # 只可视化第一个环境，避免过于混乱
            })
            
            # 选择要可视化的环境
            env_id = traj_config.get("visualize_env_id", 0)
            if env_id >= self.num_envs:
                env_id = 0
            
            # 获取该环境的轨迹点
            trajectory = self.current_trajectories[env_id]  # (num_waypoints, 3)
            
            # 构建线段：从每个点到下一个点
            start_points = []
            end_points = []
            
            for i in range(len(trajectory) - 1):
                start_p = trajectory[i]
                end_p = trajectory[i + 1]
                
                start_points.append(tuple(start_p.cpu().numpy().astype(float)))
                end_points.append(tuple(end_p.cpu().numpy().astype(float)))
            
            # 如果是闭合轨迹，连接最后一个点到第一个点
            if traj_config.get("closed_loop", True):
                start_points.append(tuple(trajectory[-1].cpu().numpy().astype(float)))
                end_points.append(tuple(trajectory[0].cpu().numpy().astype(float)))
            
            # 绘制轨迹线
            num_lines = len(start_points)
            colors = [traj_config["color"]] * num_lines
            widths = [traj_config["line_width"]] * num_lines
            
            self.debug_draw.draw_lines(start_points, end_points, colors, widths)
            
            # 可选：绘制当前目标点（box 最近的轨迹点）
            if hasattr(self, 'trajectory_closest_points') and traj_config.get("show_closest_point", True):
                closest_point = self.trajectory_closest_points[env_id]
                # 绘制一个小球标记最近点
                self.debug_draw.draw_points(
                    [tuple(closest_point.cpu().numpy().astype(float))],
                    [(1.0, 0.0, 0.0, 1.0)],  # 红色
                    [10.0]  # 点的大小
                )
            
        except Exception as e:
            print(f"Warning: Failed to update trajectory visualization: {e}")
            self.trajectory_visualization_enabled = False
    # ------------------------------ End of Trajectory Visualization setup -------------------------------
    
    def _reset_idx(self, idx: torch.Tensor):
        """Reset the environment at the given indices.

        Note:
            This function inherits from :meth:`isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv._reset_idx`.
            This is done because SKRL requires the "episode" key in the extras dict to be present in order to log.
        Args:
            idx (torch.Tensor): Indices of the environments to reset.
        """
        super()._reset_idx(idx)
        if hasattr(self, "reached_box_flags"):
            self.reached_box_flags[idx] = False
        
        if hasattr(self, '_prev_trajectory_progress'):
            self._prev_trajectory_progress[idx] = 0.0
        

        if hasattr(self, '_prev_trajectory_distance'):
            self._prev_trajectory_distance[idx] = 0.0
        

        if hasattr(self, '_prev_closest_points'):
            self._prev_closest_points[idx] = 0.0
        

        if hasattr(self, 'flag_trajectory_finished'):
            self.flag_trajectory_finished[idx] = False
        
        if hasattr(self, 'closest_point_index'):
            self.closest_point_index[idx] = 0

        # Reset previous distances for distance decrease reward
        self._prev_robot_box_dist[idx] = get_distance(self, "robot", "box_1")[idx]
        goal_pos = self.current_trajectories[:, -1, :3]  # (num_envs, 3)
        robot_pos = self.scene["box_1"].data.body_link_state_w[:, 0, :3]  # (num_envs, 3)
        self._prev_box_goal_dist[idx] = torch.norm(goal_pos - robot_pos, dim=1)[idx]
        # Done this way because SKRL requires the "episode" key in the extras dict to be present in order to log.
        self.extras["episode"] = self.extras["log"]
        
        # get new trajectories for the reset environments
        # box_pos = self.scene["box_1"].data.body_link_state_w[:, 0, :3][idx]  # get the 3D position of the box
        # new_trajectories = self.trajectory_generator.generate_trajectories(start_pos=box_pos, num_envs=len(idx), env_ids=idx)
        # if self.current_trajectories is None:
        #     self.current_trajectories = torch.zeros(
        #         self.num_envs,
        #         self.cfg.trajectory_cfg.num_waypoints,
        #         3,
        #         device=self.device
        #     ) # initialize if None
        # self.current_trajectories[idx] = new_trajectories
        
        if hasattr(self, 'trajectory_progress'):
            self.trajectory_progress[idx] = 0.0  # reset progress index for these envs
        
        if hasattr(self, 'trajectory_closest_points'):
            self.trajectory_closest_points[idx] = 0.0
        
        if hasattr(self, 'trajectory_distance'):
            self.trajectory_distance[idx] = 0.0

    # This function is reimplemented to make visualization less laggy
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Update FOV visualization.
        8. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self.global_step_counter += 1
        # process actions
        self.action_manager.process_action(action)

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # perform rendering if gui is enabled
            if self.sim.has_gui():
                self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        
        # -- update trajectory progress
        box_pos = self.scene["box_1"].data.body_link_state_w[:, 0, :3]  # (num_envs, 3)
        closest_points, closest_indices, distances, progresses = self.trajectory_generator.get_closest_point_on_trajectory(
            box_pos=box_pos,
            trajectories=self.current_trajectories
        )
        
        self.closest_point_index = closest_indices
        self.trajectory_progress = progresses
        self.trajectory_distance = distances
        self.trajectory_closest_points = closest_points
        
        # self.trajectory_tracking_distance = distances   # distance bias for reward computation
        self.extras['trajectory_progress'] = progresses
        self.extras['trajectory_distance'] = distances
        self.extras['trajectory_closest_points'] = closest_points
        
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        
        # -- update prev distance buffers for distance decrease reward
        self._prev_trajectory_progress = progresses.clone()
        self._prev_closest_points = closest_points.clone()
        self._prev_trajectory_distance = distances.clone()
        
        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- apply curriculum learning
        if hasattr(self, 'curriculum_manager'):
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.curriculum_manager.compute(env_ids=env_ids)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # -- update FOV visualization
        if self.fov_visualization_enabled:
            self._update_fov_visualization()
        
        if self.trajectory_visualization_enabled:
            self._update_trajectory_visualization()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

