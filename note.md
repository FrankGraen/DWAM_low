🗓️ 2025-11-29 🕒 01:00:00 - 03:18:00
1. 考虑扩大半径范围，降低 `out_of_bound` 对实验的影响；或者尝试优化生成轨迹的算法，保证轨迹生成在合理范围内。
- 🛠️ 修改了轨迹生成的逻辑，直线按照向着圆心的方向生成。
- 任务覆盖全面：函数分为三个阶段：

    - 接近箱子阶段：robot_angle_to_box角度对齐）、reached_box（到达箱子）、distance_to_box_reward（距离减少，代码中有但未在列表中列出）。
    - 推动箱子阶段：trajectory_distance_reward（距离轨迹）、trajectory_progress_reward（进度）、trajectory_trajectory_velocity_alignment_reward（速度对齐）、slow_down_near_finish（接近终点减速）。
    - 完成阶段：trajectory_progress_finish（完成轨迹）、reached_box（稀疏奖励）。
这符合任务目标：先接近箱子，再沿直线轨迹推动箱子到目标。

2. 考虑将`robot_angle_to_box`设置为只在第一阶段发挥的奖励，避免在推动箱子阶段产生干扰。

3. 对参数进行进一步调整：
    - `trajectory_distance_reward` 权重增加到0.01
    - `trajectory_progress_reward` 权重增加到0.015

运行代码：
```bash
    python scripts/train.py --num_envs=8192 --enable_cameras --headless
```

测试代码：
```bash
    python scripts/play.py --num_envs=1 --enable_cameras --checkpoint="/home/wzx/Documents/DWAM/checkpoints/agent_400000.pt"
```

🗓️ 2025-11-29 🕒 13:20:00 - 14:00:00
1. 考虑对进度进行稀疏奖励的调整，即设置进度里程碑，在达到特定进度点时给予额外奖励，鼓励推动箱子完成任务。

🛠️ 新增了奖励函数`trajectory_following_reward_milestone`，用于给定历程奖励。
    - 其中，里程碑点设置为0.3、0.6、0.9，对应奖励分别为1.0，权重均设置为10.0。
    - 每个函数单独控制一个里程碑点，确保奖励的独立性和可调节性。
（已经初步在手控程序中验证了奖励参数的可行性）
2. 调整了在reward函数中更新先前进度的逻辑，确保在计算历程奖励时使用的是上一个时间步的进度值。