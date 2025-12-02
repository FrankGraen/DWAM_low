# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates robots with keyboard control, third-person, and first-person cameras in a multi-environment setup.

.. code-block:: bash

    # Usage
    python scripts/control.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates robots with keyboard control and third-person/first-person cameras in multiple environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--save_cam", type=bool, default=False, help="Whether to save ego-centric camera images.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import carb.input
import omni.appwindow
import omni.kit.viewport.utility
import sys
import os

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # to include parent directory for imports
# Import the keyboard control and camera logic modules
import utils
from envs.robot_env_cfg import RobotEnvCfg
from envs.robot_env import RobotEnv
import robot_keyboard_ctrl
from camera_logic import create_cameras, update_third_person_camera, save_images_grid

from envs.robots import turtle_se
usePID = False  # Set to True to use PID controller for yaw correction

class PIDController:
    """Simple PID controller for yaw correction."""
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error: float) -> float:
        """Compute PID control output."""
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    
def run_simulator(env):
    """Run the simulator using the RL environment, but action细节保留原robot.py的vel_cmd、PID、逆运动学等逻辑。"""
    import numpy as np
    from envs.robots import turtle_se
    count = 0
    num_envs = env.num_envs
    robot_keyboard_ctrl.init_base_vel_cmd(num_envs)
    camera_view_state = [0] * num_envs
    sim_dt = env.physics_dt
    cameras = create_cameras(num_envs)

    # Create output directory to save images
    if args_cli.save_cam:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "output")
        os.makedirs(output_dir, exist_ok=True)

    # Initialize PID controllers for each environment
    pid_controllers = [PIDController(kp=10.0, ki=0.1, kd=0.5, dt=sim_dt) for _ in range(num_envs)]
    target_yaw = [0.0] * num_envs
    
    # Setup keyboard input
    system_input = carb.input.acquire_input_interface()
    for env_idx in range(num_envs):
        system_input.subscribe_to_keyboard_events(
            omni.appwindow.get_default_app_window().get_keyboard(),
            lambda event, idx=env_idx: robot_keyboard_ctrl.sub_keyboard_event(event, idx)
        )
    # Setup viewport
    viewport_api = omni.kit.viewport.utility.get_active_viewport()
    if viewport_api:
        viewport_api.set_active_camera("/World/envs/env_0/Camera")

    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 2000 == 0:
                sim_time = 0.0
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                actions = torch.zeros((num_envs, 2), device=env.device)
            for env_idx in range(num_envs):
                vel_cmd = robot_keyboard_ctrl.get_base_vel_cmd(env_idx) * 2.6  # [vx, vy, wz]
                # Velocity command in robot's local frame
                root_vel = torch.zeros((1, 6), device=env.device)
                root_vel[0, 0] = vel_cmd[0]  # linear x velocity
                root_vel[0, 5] = vel_cmd[2]  # angular z velocity

                # -------------------- PID Control for Yaw Correction --------------------
                if usePID:
                    # Get current yaw from robot's quaternion
                    quat = env.scene["robot"].data.root_state_w[env_idx, 3:7].detach().cpu().numpy()
                    rot = torch.tensor(quat[[1, 2, 3, 0]], dtype=torch.float32)  # xyzw
                    current_yaw = float(torch.atan2(2.0 * (rot[0] * rot[1] + rot[2] * rot[3]), 1.0 - 2.0 * (rot[1]**2 + rot[2]**2)))
                    # Update target yaw when robot is turning
                    if abs(vel_cmd[2]) > 1e-3:  # If angular velocity command is non-zero, update target yaw
                        target_yaw[env_idx] = current_yaw
                    else:  # If moving straight, use PID to correct yaw
                        yaw_error = target_yaw[env_idx] - current_yaw
                        # Normalize yaw error to [-pi, pi]
                        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
                        correction = pid_controllers[env_idx].compute(yaw_error)
                        root_vel[0, 5] += correction  # Add correction to angular velocity
                # ----------------------------------------------------------------------

                # Convert to wheel velocities
                wheel_vels = turtle_se.inverse_kinematics_tensor_from_root_vel(root_vel)
                actions[env_idx] = wheel_vels[0]

                # -------------------- Convert the velocity command to world frame  --------------------------
                quat = env.scene["robot"].data.root_state_w[env_idx, 3:7].detach().cpu().numpy()
                rot = torch.tensor(quat[[1,2,3,0]], dtype=torch.float32)  # xyzw
                yaw = float(torch.atan2(2.0*(rot[0]*rot[1] + rot[2]*rot[3]), 1.0 - 2.0*(rot[1]**2 + rot[2]**2)))
                yaw_mat = torch.tensor([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0,           0.0,        1.0]
                ], dtype=torch.float32)
                vel_body = torch.tensor([vel_cmd[0], 0.0, 0.0], dtype=torch.float32)
                vel_world = yaw_mat @ vel_body  # Rotate the velocity command to world frame
                root_vel_world = torch.zeros((1, 6), device=env.device)
                root_vel_world[0, 0:2] = vel_world[:2]
                root_vel_world[0, 5] = vel_cmd[2]
                env.scene["robot"].write_root_velocity_to_sim(root_vel_world, env_ids= torch.tensor([env_idx], dtype=torch.int32, device=env.device))
                # --------------------------------------------------------------------------------------------

                # if camera_view_state[env_idx] == 0: #FIXME: Camera need to be fixed
                #     update_third_person_camera(env.scene, env_idx)

                # if camera_view_state[env_idx] == 1:
                #     vp_name = f"Viewport_{env_idx}" if env_idx > 0 else "Viewport"
                #     vp_api = omni.kit.viewport.utility.get_viewport_from_window(omni.appwindow.get_default_app_window(), vp_name)
                #     if vp_api:
                #         cam_path = f"/World/envs/env_{env_idx}/Robot/base_footprint/base_link/front_cam_1"
                #         vp_api.set_active_camera(cam_path)

            obs, rew, terminated, truncated, info = env.step(actions)
            sim_time += sim_dt
            env.scene.update(dt=sim_dt)
            
            # Print debug information
            if count % 1 == 0:
                env.print_rewards()
                env.print_observations()
            # if count % 1 == 0:
                # print(f"{'[Env 0]: Robot obs:' if 'policy' in obs else 'No policy obs'}, dis_box:{obs.get('policy', torch.zeros(1, 1))[0][2:3]}, ang_box:{obs.get('policy', torch.zeros(1, 1))[0][3:4]}")
                # print(f"{'[Env 0]: Robot obs:' if 'policy' in obs else 'No policy obs'}, dis_box_goal:{obs.get('policy', torch.zeros(1, 1))[0][4:5]}, ang_goal:{obs.get('policy', torch.zeros(1, 1))[0][5:6]}")
                # print(f"[Env 0]: Reward: {rew[0].item():.10f}, reached_box:{rew[0][2:3].item():.10f}")
                # print(f"{info['episode']['Episode_Reward/reached_box']}, reached_box_flags:{env.reached_box_flags[0].item()}")
                # print(f"{info['episode']}, reached_box_flags:{env.reached_box_flags[0].item()}")

            # save every 10th image (for visualization purposes only)
            # note: saving images will slow down the simulation
            if args_cli.save_cam:
                if count % 10 == 0:
                    # save generated Depth images
                    depth_images = [
                        env.scene["camera"].data.output["distance_to_image_plane"][0, ..., 0],
                    ]
                    save_images_grid(
                        depth_images,
                        cmap="turbo",
                        title="Depth Image: Cam0",
                        filename=os.path.join(output_dir, "distance_to_camera", f"{count:04d}.jpg"),
                    )

                    # save tiled RGB images
                    tiled_images = env.scene["camera"].data.output["rgb"]
                    save_images_grid(
                        tiled_images,
                        title="Tiled RGB Image: Cam0",
                        filename=os.path.join(output_dir, "tiled_rgb", f"{count:04d}.jpg"),
                    )

            count += 1
    env.close()

def main():
    """Main function."""
    # create environment configuration
    env_cfg = RobotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = RobotEnv(cfg=env_cfg)

    print("[INFO]: Setup complete. Starting simulation...")
    run_simulator(env)

if __name__ == "__main__":
    main()
    simulation_app.close()