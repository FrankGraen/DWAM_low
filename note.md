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


🗓️ 2025-12-5
1. 问题：如果采用goal引导追踪轨迹的方式，会导致rbt在推箱子时一步过大从而跳过目标点，无法及时更新goal位置，进而影响奖励计算。
🛠️ 解决方案：考虑扩大轨迹的采样点，保证一个step只完成一个goal的目标，从而便于目标的维护。

2. 问题：如果采用让rbt学习逐步跟随目标点的方式，会导致rbt在推箱子的时难以学习到有效的策略，前期收敛很困难。
🛠️ 解决方案：考虑使用课程学习（Curriculum Learning），开始时任务是push_box_to_goal，之后逐步增加轨迹的采样点，引导rbt学会连续轨迹跟踪中的局部规划问题。

3. 综合：我们的input是 box_local, goal_local, trajectory_future，目的是让rbt学会根据box和goal的位置关系，结合未来轨迹信息，规划出合理的动作序列（local motion planner 而不是 global planner）。
