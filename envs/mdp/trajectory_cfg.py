import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal
TRAJECTORY_TYPE = "s_curve"

@dataclass
class TrajectoryConfig:
    """轨迹配置"""
    trajectory_type: Literal["line", "circle", "s_curve", "square"] = TRAJECTORY_TYPE    # 直线、圆形、S形、方形
    trajectory_length: float = 5.0  # 轨迹总长度
    num_waypoints: int = 100  # 轨迹采样点数量
    tolerance: float = 0.1  # 允许偏离轨迹的距离
    
    # 不同轨迹的参数
    circle_radius: float = 0.8
    line_length: float = 2.0
    s_curve_amplitude: float = 0.6
    square_side: float = 1.2
    
    # 最终的参数，用于其他模块调用
    trajectory_params: float = 0.0
    if trajectory_type == "line":
        trajectory_params = line_length
    elif trajectory_type == "circle":
        trajectory_params = circle_radius
    elif trajectory_type == "s_curve":
        trajectory_params = s_curve_amplitude
    elif trajectory_type == "square":
        trajectory_params = square_side


class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg
        
    def generate_trajectories(self, start_pos: torch.Tensor):
        """
        为每个环境生成轨迹
        
        Args:
            start_pos: (num_envs, 3) 箱子的起始位置
        
        Returns:
            trajectories: (num_envs, num_waypoints, 3) 轨迹点
        """
        device = start_pos.device
        
        # 根据配置的轨迹类型生成
        if self.cfg.trajectory_type == "line":
            return self._generate_line_trajectory(start_pos, device)
        elif self.cfg.trajectory_type == "circle":
            return self._generate_circle_trajectory(start_pos, device)
        elif self.cfg.trajectory_type == "s_curve":
            return self._generate_s_curve_trajectory(start_pos, device)
        elif self.cfg.trajectory_type == "square":
            return self._generate_square_trajectory(start_pos, device)
        else:
            raise ValueError(f"Unknown trajectory type: {self.cfg.trajectory_type}")
    
    def _generate_line_trajectory(self, start_pos: torch.Tensor, device):
        """生成直线轨迹"""
        num_envs = start_pos.shape[0]
        num_points = self.cfg.num_waypoints
        
        # 大致方向向着原点，在这个基础上添加随机扰动
        to_orign = - start_pos[:, :2] # (num_envs, 2)
        base_direction = to_orign / (torch.norm(to_orign, dim=-1, keepdim=True) + 1e-6)  # 单位向量\
        
        angle_noise = (torch.rand((num_envs, 1), device=device) - 0.5) * np.pi / 4  # ±22.5度
        
        # 最终方向
        cos_theta = torch.cos(angle_noise)  # (num_envs, 1)
        sin_theta = torch.sin(angle_noise)  # (num_envs, 1)
        
        directions = torch.zeros((num_envs, 2), device=device)
        directions[:, 0:1] = base_direction[:, 0:1] * cos_theta - base_direction[:, 1:2] * sin_theta
        directions[:, 1:2] = base_direction[:, 0:1] * sin_theta + base_direction[:, 1:2] * cos_theta
        
        # 生成轨迹点
        t = torch.linspace(0, self.cfg.line_length, num_points, device=device)
        t = t.unsqueeze(0).repeat(num_envs, 1)  # (num_envs, num_points)
        
        trajectories = torch.zeros((num_envs, num_points, 3), device=device)
        trajectories[:, :, 0] = start_pos[:, 0:1] + t * directions[:, 0:1]
        trajectories[:, :, 1] = start_pos[:, 1:2] + t * directions[:, 1:2]
        trajectories[:, :, 2] = start_pos[:, 2:3]  # 保持高度不变
        
        return trajectories
    
    def _generate_circle_trajectory(self, start_pos: torch.Tensor, device):
        """生成圆形轨迹"""
        num_envs = start_pos.shape[0]
        num_points = self.cfg.num_waypoints
        
        # 随机圆心（在起点附近）
        center = start_pos.clone()
        center[:, 0] += self.cfg.circle_radius
        
        # 生成圆形轨迹
        angles = torch.linspace(0, 2 * np.pi, num_points, device=device)
        angles = angles.unsqueeze(0).repeat(num_envs, 1)
        
        trajectories = torch.zeros((num_envs, num_points, 3), device=device)
        trajectories[:, :, 0] = center[:, 0:1] + self.cfg.circle_radius * torch.cos(angles)
        trajectories[:, :, 1] = center[:, 1:2] + self.cfg.circle_radius * torch.sin(angles)
        trajectories[:, :, 2] = start_pos[:, 2:3]
        
        return trajectories
    
    def _generate_s_curve_trajectory(self, start_pos: torch.Tensor, device):
        """生成偏向圆心方向的 S 型曲线轨迹"""
        num_envs = start_pos.shape[0]
        num_points = self.cfg.num_waypoints
        
        # ===== 1. 主方向：指向圆心 =====
        to_origin = - start_pos[:, :2]   # (num_envs, 2)
        base_direction = to_origin / (torch.norm(to_origin, dim=-1, keepdim=True) + 1e-6)  # 单位向量
        
        # ===== 2. 添加方向扰动（可选） =====
        angle_noise = (torch.rand((num_envs, 1), device=device) - 0.5) * np.pi / 6  # ±15°
        cos_theta = torch.cos(angle_noise)
        sin_theta = torch.sin(angle_noise)

        directions = torch.zeros((num_envs, 2), device=device)
        directions[:, 0:1] = base_direction[:, 0:1] * cos_theta - base_direction[:, 1:2] * sin_theta
        directions[:, 1:2] = base_direction[:, 0:1] * sin_theta + base_direction[:, 1:2] * cos_theta

        # ===== 3. 垂直方向向量（用于 S 形波动）=====
        # 若主方向为 (dx, dy)，垂直方向可取 (-dy, dx)
        perp = torch.stack([-directions[:, 1], directions[:, 0]], dim=-1)  # (num_envs, 2)
        perp = perp / (torch.norm(perp, dim=-1, keepdim=True) + 1e-6)

        # ===== 4. 参数 t =====
        t = torch.linspace(0, 1, num_points, device=device)
        t = t.unsqueeze(0).repeat(num_envs, 1)  # (num_envs, num_points)

        # ===== 5. 构建 S 形波动 =====
        # 正弦从 0 到 2π
        s = torch.sin(t * 2 * np.pi) * self.cfg.s_curve_amplitude  # (num_envs, num_points)

        # ===== 6. 生成轨迹 =====
        trajectories = torch.zeros((num_envs, num_points, 3), device=device)

        # 主方向前进：t * length
        main_step = t * self.cfg.line_length  # (num_envs, num_points)

        trajectories[:, :, 0] = (
            start_pos[:, 0:1] + 
            main_step * directions[:, 0:1] + 
            s * perp[:, 0:1]
        )

        trajectories[:, :, 1] = (
            start_pos[:, 1:2] +
            main_step * directions[:, 1:2] +
            s * perp[:, 1:2]
        )

        # 高度保持不变
        trajectories[:, :, 2] = start_pos[:, 2:3]

        return trajectories
    
    def _generate_square_trajectory(self, start_pos: torch.Tensor, device):
        """生成方形轨迹"""
        num_envs = start_pos.shape[0]
        points_per_side = self.cfg.num_waypoints // 4
        
        side = self.cfg.square_side
        
        # 四条边的轨迹
        trajectories = []
        
        # 边1: 向右
        t = torch.linspace(0, side, points_per_side, device=device)
        edge1 = torch.zeros((num_envs, points_per_side, 3), device=device)
        edge1[:, :, 0] = start_pos[:, 0:1] + t.unsqueeze(0)
        edge1[:, :, 1] = start_pos[:, 1:2]
        edge1[:, :, 2] = start_pos[:, 2:3]
        
        # 边2: 向前
        edge2 = torch.zeros((num_envs, points_per_side, 3), device=device)
        edge2[:, :, 0] = start_pos[:, 0:1] + side
        edge2[:, :, 1] = start_pos[:, 1:2] + t.unsqueeze(0)
        edge2[:, :, 2] = start_pos[:, 2:3]
        
        # 边3: 向左
        edge3 = torch.zeros((num_envs, points_per_side, 3), device=device)
        edge3[:, :, 0] = start_pos[:, 0:1] + side - t.unsqueeze(0)
        edge3[:, :, 1] = start_pos[:, 1:2] + side
        edge3[:, :, 2] = start_pos[:, 2:3]
        
        # 边4: 向后
        edge4 = torch.zeros((num_envs, points_per_side, 3), device=device)
        edge4[:, :, 0] = start_pos[:, 0:1]
        edge4[:, :, 1] = start_pos[:, 1:2] + side - t.unsqueeze(0)
        edge4[:, :, 2] = start_pos[:, 2:3]
        
        trajectories = torch.cat([edge1, edge2, edge3, edge4], dim=1)
        
        return trajectories
    
    def get_closest_point_on_trajectory(self, box_pos: torch.Tensor, trajectories: torch.Tensor):
        """
        找到箱子在轨迹上的最近点
        
        Args:
            box_pos: (num_envs, 3) 箱子当前位置
            trajectories: (num_envs, num_waypoints, 3) 轨迹
        
        Returns:
            closest_points: (num_envs, 3) 最近的轨迹点
            closest_point_indices: (num_envs,) 最近点的索引
            distances: (num_envs,) 到轨迹的距离
            progress: (num_envs,) 轨迹进度 [0, 1]
        """
        # 计算箱子到所有轨迹点的距离
        diff = trajectories - box_pos.unsqueeze(1)  # (num_envs, num_waypoints, 3)
        distances_to_points = torch.norm(diff[:, :, :2], dim=-1)  # 只考虑xy平面
        
        # 找到最近的点
        closest_indices = torch.argmin(distances_to_points, dim=-1)  # (num_envs,)
        
        # 提取最近点
        batch_indices = torch.arange(trajectories.shape[0], device=trajectories.device)
        closest_points = trajectories[batch_indices, closest_indices]
        
        # 计算距离
        min_distances = distances_to_points[batch_indices, closest_indices]
        
        # 计算进度
        progress = closest_indices.float() / (self.cfg.num_waypoints - 1)
        
        return closest_points, closest_indices, min_distances, progress