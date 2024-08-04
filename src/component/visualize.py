from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from open3d.visualization import Visualizer

from src.data.dataset import BaseDataset


class PcdVisualizer:

    def __init__(self, intrinsic_matrix: NDArray[np.int32], view_scale=1.0):
        self.o3d_vis = Visualizer()
        self.o3d_vis.create_window(
            window_name="Complete Point Cloud", width=1600, height=1200, visible=True
        )
        # Set camera parameters for following the camera pose
        self.view_control = self.o3d_vis.get_view_control()
        self.camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=int(1200 * view_scale),
            height=int(680 * view_scale),
            fx=intrinsic_matrix[0, 0] * view_scale,
            fy=intrinsic_matrix[1, 1] * view_scale,
            cx=intrinsic_matrix[0, 2] * view_scale,
            cy=intrinsic_matrix[1, 2] * view_scale,
        )
        self.camera_params.intrinsic = intrinsic
        render_option = Path(__file__).parents[2] / "data" / "render_option.json"
        self.o3d_vis.get_render_option().load_from_json(render_option.as_posix())

    def update_render(
        self,
        new_pcd: NDArray[np.float32],
        estimate_pose: NDArray[np.float32],
        new_color: NDArray[np.float32] | None = None,
        down_sample: float = 0.05,
    ):
        # new_pcd = o3d.utility.Vector3dVector(new_pcd[:, :3])
        new_pcd = o3d.utility.Vector3dVector(new_pcd)
        pcd_o3d = o3d.geometry.PointCloud(new_pcd)
        if new_color is not None:
            pcd_o3d.colors = o3d.utility.Vector3dVector(new_color)

        # pcd_o3d = pcd_o3d.random_down_sample(down_sample)

        pcd_o3d.transform(estimate_pose)
        self.o3d_vis.add_geometry(pcd_o3d)
        self.o3d_vis.update_geometry(pcd_o3d)
        self._follow_camera(estimate_pose)
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def _follow_camera(self, c2w: NDArray[np.float64]):
        """Adjust the view control to follow a series of camera transformations."""

        self.camera_params.extrinsic = np.linalg.inv(
            c2w
        )  # Convert c2w to camera intrinsic
        self.view_control.convert_from_pinhole_camera_parameters(
            self.camera_params, allow_arbitrary=True
        )

    def close(self):
        self.o3d_vis.destroy_window()

    def vis_trajectory(
        self,
        gt_poses: list[NDArray],
        estimated_poses: list[NDArray],
        downsampling_resolution: float,
        fps: float,
    ) -> None:
        """Visualize the camera trajectory in 2D space."""
        gt_traj = np.array([pose[:3, 3] for pose in gt_poses])
        icp_traj = np.array([pose[:3, 3] for pose in estimated_poses])
        plt.clf()
        plt.title(f"Downsample ratio {downsampling_resolution}\nfps : {fps:.2f}")
        plt.plot(icp_traj[:, 0], icp_traj[:, 1], label="g-icp trajectory", linewidth=3)
        plt.legend()
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], label="ground truth trajectory")
        plt.legend()
        plt.axis("equal")
        plt.pause(0.01)


def visualize_dataset(data_set: BaseDataset):

    vis = PcdVisualizer(intrinsic_matrix=data_set.K)
    for i, rgbd_image in enumerate(data_set):

        print(f"Processing image {i + 1}/{len(data_set)}...")
        vis.update_render(
            rgbd_image.points.cpu().numpy(),
            rgbd_image.pose.cpu().numpy(),
            new_color=rgbd_image.colors.cpu().numpy(),
            down_sample=0.01,
        )


def visualize_trajectory(data_set: BaseDataset):
    """vis 3d trajectory"""
    poses = [
        rgbd_image.pose.cpu().numpy()
        for rgbd_image in data_set
        if rgbd_image.pose is not None
    ]
    translations = [pose[:3, 3] for pose in poses]

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    trajs = np.array(translations)
    ax.plot(trajs[:, 0], trajs[:, 1], trajs[:, 2], "-b")  # Blue line for trajectory

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.title("Camera Trajectory")
    plt.show()


def multi_vis_img(imgs: list[NDArray]) -> None:
    """Visualize multiple images in a grid layout."""
    n = len(imgs)  # Number of depth images
    cols = 2  # Number of columns in subplot grid
    rows = (n + cols - 1) // cols  # Calculate required rows, round up division

    plt.figure(figsize=(15, 5 * rows))  # Adjusted figsize based on content

    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)  # Dynamic subplot positioning
        plt.imshow(img)
        # plt.title(f"Image:{img}")
        plt.axis("off")

    plt.tight_layout()  # Adjust layout so plots do not overlap
    plt.show()


def vis_depth_filter(depths: list[NDArray]) -> None:
    """Visualize the depth images as colormap in a grid layout."""
    multi_vis_img([depth_to_colormap(depth) for depth in depths])


def depth_to_colormap(depth_image: NDArray):
    # Normalize and convert the depth image to an 8-bit format, and apply a colormap
    depth_normalized = (depth_image - np.min(depth_image)) / (
        np.max(depth_image) - np.min(depth_image)
    )
    depth_8bit = np.uint8(depth_normalized * 255)
    depth_colormap = plt.cm.jet(depth_8bit)  # Using matplotlib's colormap
    return depth_colormap


def show_image(color: NDArray, depth: NDArray):
    # Plot color image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if color.max() > 1.0:
        color = color / 255.0
    plt.imshow(color)
    plt.title("Color Image")
    plt.axis("off")

    # Plot depth image with colormap
    plt.subplot(1, 2, 2)
    depth_colormap = depth_to_colormap(depth)
    plt.imshow(depth_colormap)
    plt.title("Depth Image (Colormap)")
    plt.axis("off")

    # Add colorbar
    plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=plt.gca())

    plt.show()


def visualize_point_cloud(points: torch.Tensor, colors: torch.Tensor):
    """
    Visualize point cloud data using Open3D.

    Parameters:
    points (torch.Tensor): Tensor of point coordinates with shape (N, 3).
    colors (torch.Tensor): Tensor of colors with shape (N, 3), values range from 0 to 1.

    """
    # Ensure input has correct shape
    assert points.shape[1] == 3, "Point coordinates should have shape (N, 3)"
    assert colors.shape[1] == 3, "Colors should have shape (N, 3)"
    assert (
        points.shape[0] == colors.shape[0]
    ), "Number of points and colors should be the same"

    # Convert to numpy arrays
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
