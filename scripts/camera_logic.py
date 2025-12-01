import numpy as np
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.viewports as viewports
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import matplotlib.pyplot as plt
import torch
import os

def create_cameras(num_envs: int):
    """Create third-person and first-person cameras for each environment."""
    cameras = []

    for env_idx in range(num_envs):
        # Third-person
        camera_path = f"/World/envs/env_{env_idx}/Camera"
        prim = prim_utils.get_prim_at_path(camera_path)
        if not prim.IsValid():
            prim_utils.create_prim(camera_path, "Camera", translation=np.array([0.0, 0.0, 5.0]))
            prim = prim_utils.get_prim_at_path(camera_path)
            prim.GetAttribute("focalLength").Set(24.0)
            prim.GetAttribute("focusDistance").Set(4.0)
            prim.GetAttribute("fStop").Set(0.0)

        # First-person
        fp_camera_path = f"/World/envs/env_{env_idx}/Robot/base_footprint/base_link/front_cam_1"
        if not prim_utils.get_prim_at_path(fp_camera_path).IsValid():
            prim_utils.create_prim(fp_camera_path, "Camera")

        camera = Camera(
            prim_path=fp_camera_path,
            translation=np.array([0.035, 0.0, 0.18]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            frequency=60.0,
            resolution=(640, 480)
        )
        camera.initialize()
        camera.set_focal_length(1.5)
        camera.set_clipping_range(0.01, 100.0)
        cameras.append(camera)

    return cameras

def update_third_person_camera(scene, env_idx: int):
    """Update third-person camera to follow the robot."""
    robot_position = scene["robot"].data.root_state_w[env_idx, :3].cpu().numpy()
    robot_orientation = scene["robot"].data.root_state_w[env_idx, 3:7].cpu().numpy()

    rotation = R.from_quat([robot_orientation[1], robot_orientation[2], robot_orientation[3], robot_orientation[0]])
    yaw = rotation.as_euler('zyx')[0]
    yaw_rotation = R.from_euler('z', yaw).as_matrix()

    camera_path = f"/World/envs/env_{env_idx}/Camera"
    eye = yaw_rotation.dot(np.asarray([-3.5, 0.0, 3.0])) + robot_position
    target = robot_position

    viewports.set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path=camera_path
    )

def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    
    # Ensure axes is a list of Axes objects
    if nrow == 1 and ncol == 1:
        axes = [axes]  # Single Axes object to list
    elif nrow == 1 or ncol == 1:
        axes = axes.ravel()  # Convert 1D array to flat list
    else:
        axes = axes.flatten()  # Convert 2D array to flat list

    # plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])
    # remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)
    # set title
    if title:
        plt.suptitle(title)

    # adjust layout to fit the title
    plt.tight_layout()
    # save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # close the figure
    plt.close()