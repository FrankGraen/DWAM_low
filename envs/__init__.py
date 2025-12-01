import os
import gymnasium as gym
from .robot_env_cfg import RobotEnvCfg
from .robot_env import RobotEnv
from learning.skrl import get_agent

# ROBOT_LAB_ENVS_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'learning')

gym.register(
    id="BoxPushEnv-v0",
    entry_point='envs:RobotEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobotEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt",
        "get_agent_fn": get_agent,
    }
)