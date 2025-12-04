import torch
def trajectory_progress_finish_reward(
    num_envs: int,
    reward_amount: float = 0.2,
    max_episode_length: int = 1000,
    episode_length_buf: torch.Tensor = None,
    env_finish_goals: torch.Tensor = None,
    env_total_goals: torch.Tensor = None,
    env_flag_trajectory_finished: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reward for finishing the trajectory by reaching the target.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        reward_amount (float): Reward amount for finishing the trajectory.

    Returns:
        torch.Tensor: Reward for finishing the trajectory, shape (num_envs,).
    """
    reward = torch.zeros(num_envs, device='cpu')
    
    finish_goals = env_finish_goals #(num_envs,)
    total_goals = env_total_goals #(num_envs,)
    finished_mask = finish_goals == total_goals # (num_envs,)
    just_finished_mask = finished_mask & (~env_flag_trajectory_finished)
    
    # 计算时间缩放奖励
    remaining_steps = max_episode_length - episode_length_buf
    reward_scale = remaining_steps / (max_episode_length + 1e-8)
    reward[just_finished_mask] = reward_amount * reward_scale[just_finished_mask]
    
    env_flag_trajectory_finished |= just_finished_mask
    return reward, env_flag_trajectory_finished

finish_goals = torch.tensor([1, 2, 3, 3])
total_goals = 3
episode_length_buf = torch.tensor([100, 200, 300, 400])
env_flag_trajectory_finished = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
num_envs = 4
reward, env_flag_trajectory_finished = trajectory_progress_finish_reward(
    num_envs=num_envs,
    reward_amount=0.2,
    max_episode_length=1000,
    episode_length_buf=episode_length_buf,
    env_finish_goals=finish_goals,
    env_total_goals=total_goals,
    env_flag_trajectory_finished=env_flag_trajectory_finished,
)
print(reward)  # 输出奖励
print(env_flag_trajectory_finished)  # 输出更新后的标志