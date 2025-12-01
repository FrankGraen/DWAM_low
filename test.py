# test_local_transform_simple.py
import torch
import math

# ================== 严格按照你原始代码实现的函数 ==================
def world_to_local(result, robot_pos, robot_quat, device, flatten=False):
    """
    完全 1:1 复刻你提供的代码逻辑
    result:      (B, N, ≥2)   世界坐标点，至少包含 xy
    robot_pos:   (B, 3)
    robot_quat:  (B, 4)       w,x,y,z
    """
    B = robot_pos.shape[0]

    # 完全按照你原来的写法计算 heading
    heading = torch.atan2(
        quat_apply(robot_quat, torch.tensor([0., 1., 0.], device=device).expand(B, 3))[:, 0],
        quat_apply(robot_quat, torch.tensor([1., 0., 0.], device=device).expand(B, 3))[:, 0]
    )  # (B,)

    c, s = heading.cos(), heading.sin()
    Rot = torch.stack([
            torch.stack([ c, -s], dim=1),   # 第0行: [cos, -sin]
            torch.stack([ s,  c], dim=1)    # 第1行: [sin,  cos]
        ], dim=1)
    print("Rotation matrix Rot shape:", Rot.shape)  # (B, 2, 2)
    local_result_xy = torch.bmm(result[..., :2] - robot_pos[:, :2].unsqueeze(1), Rot)   # (B, N, 2)

    if flatten:
        return local_result_xy.reshape(B, -1)
    else:
        return local_result_xy


# 四元数旋转向量（和 Isaac Gym / Warp 完全一致的实现）
def quat_apply(q, v):
    # 标准、正确、高效的四元数旋转向量实现
    xyz = q[:, 1:4]      # (B, 3)
    w   = q[:, 0:1]      # (B, 1)
    t   = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


# ============================== main 测试 ==============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2   # 你可以改成任意 batch size

    # ------------------- 你自己填这里 -------------------
    robot_pos = torch.tensor([
        [10.0, 10.0, 0.0],
        [30.0, 30.0, 0.0]
    ], device=device)   # (B, 3)

    # 例子：第0辆车朝向 45°，第1辆车朝向 -90°（朝向 -y）
    yaw_deg = torch.tensor([45.0, -90.0])
    yaw_rad = torch.deg2rad(yaw_deg)
    robot_quat = torch.stack([
        torch.cos(yaw_rad / 2),
        torch.zeros_like(yaw_rad),
        torch.zeros_like(yaw_rad),
        torch.sin(yaw_rad / 2)
    ], dim=1).to(device)   # (B, 4)  w,x,y,z

    # 世界坐标点（随便填几个有意义的点）
    world_points = torch.tensor([
        # batch 0
        [[10.0, 10.0, 0.0],    # 正好在车身上 → 期望 (0, 0)
         [20.0, 20.0, 0.0],    # 车正前方  ≈(3.53, 3.53) 因为 45°
         [0.0, 0.0, 0.0],    # 车正左方   →(0, 5)
         [10.0, 0.0, 0.0]],   # 随便一点
        # batch 1
        [[30.0, 30.0, 0.0],    # 车身上
         [30.0, 20.0, 0.0],    # 正前方（因为 -90° 朝 -y）
         [0.0, 0.0, 0.0],    # 正右方
         [20.0, 0.0, 0.0]]
    ], dtype=torch.float32, device=device)   # (B, N, 3)
    # ----------------------------------------------------
    print("Shape of world_points:", world_points.shape)
    local = world_to_local(world_points, robot_pos, robot_quat, device, flatten=False)
    local_flat = world_to_local(world_points, robot_pos, robot_quat, device, flatten=True)

    print("=== 局部坐标 (B, N, 2) ===")
    print(local)
    print("\n=== flatten 之后的 (B, N*2) ===")
    print(local_flat)

    # 顺便再手算一次正确答案，验证一下
    # print("\n=== 手动验证几个关键点 ===")
    # for b in range(B):
    #     yaw = yaw_deg[b].item()
    #     print(f"Batch {b}, yaw = {yaw}°")
    #     c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    #     R = torch.tensor([[c, -s], [s, c]])   # 你的 Rot 矩阵
    #     for i in range(world_points.shape[1]):
    #         dx = world_points[b, i, :2] - robot_pos[b, :2]
    #         local_manual = torch.matmul(dx, R)
    #         print(f"  点{i}: 世界 {world_points[b,i,:2].cpu().numpy()} → 局部 {local_manual.cpu().numpy()}  | 函数输出 {local[b,i].cpu().numpy()}")