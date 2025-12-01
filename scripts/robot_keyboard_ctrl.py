import torch
import carb

# Global variable to store velocity commands
base_vel_cmd_input = None

def init_base_vel_cmd(num_robots: int):
    """Initialize the velocity command tensor for the specified number of robots."""
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_robots, 3), dtype=torch.float32)

def sub_keyboard_event(event, robot_index=0) -> bool:
    """Handle keyboard events to update velocity commands."""
    global base_vel_cmd_input
    lin_vel = 0.5  # Linear velocity (m/s)
    ang_vel = 3  # Angular velocity (rad/s)
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                base_vel_cmd_input[robot_index] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'S':
                base_vel_cmd_input[robot_index] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'A':
                base_vel_cmd_input[robot_index] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'D':
                base_vel_cmd_input[robot_index] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input[robot_index].zero_()
    return True

def get_base_vel_cmd(robot_index=0) -> torch.Tensor:
    """Retrieve the velocity command for the specified robot."""
    global base_vel_cmd_input
    return base_vel_cmd_input[robot_index].clone()