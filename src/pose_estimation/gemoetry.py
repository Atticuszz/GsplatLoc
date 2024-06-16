import kornia
import torch
from torch import Tensor
import kornia.geometry.conversions as KG

from src.pose_estimation import DEVICE


def construct_full_pose(rotation, translation):
    """
    Constructs the full 4x4 transformation matrix from rotation and translation.
    Ensures that gradients are tracked for rotation and translation.
    """
    pose = torch.eye(4, dtype=torch.float64, device=DEVICE)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def project_depth(depth: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """
    Converts a depth image to a point cloud in the world coordinate system using the provided pose.

    Parameters
    ----------
    intrinsics: torch.Tensor
        The camera intrinsics with dimensions [3, 3].
    depth : torch.Tensor
        The depth image with dimensions [height, width].
    pose : torch.Tensor
        The 4x4 transformation matrix from camera to world coordinates.

    Returns
    -------
    torch.Tensor
        The converted point cloud in world coordinates with dimensions [height,width, 4].
    """
    height, width = depth.shape
    device = depth.device
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, dtype=torch.float64, device=device),
        torch.arange(height, dtype=torch.float64, device=device),
        indexing="xy",
    )

    # Transform grid to camera coordinates
    Z = depth
    X = (grid_x - intrinsics[0, 2]) * Z / intrinsics[0, 0]
    Y = (grid_y - intrinsics[1, 2]) * Z / intrinsics[1, 1]

    # Create homogenous coordinates
    ones = torch.ones_like(Z)
    pcd_camera = torch.stack((X, Y, Z, ones), dim=-1)

    # Transform to world coordinates
    pcd_world = torch.einsum("hwj,jk->hwk", pcd_camera, pose)
    return pcd_world


def unproject_depth(pcd_last, pcd_current, pose_cur):
    """
    Project two point clouds to the current camera pose and merge by selecting the minimum depth at each pixel.

    Parameters:
    pcd_last, pcd_current : torch.Tensor
        Point clouds in world coordinates, dimensions [H, W, 4] with each point as [x, y, z, 1].
    pose_cur : torch.Tensor
        The 4x4 transformation matrix from world to current camera coordinates.

    Returns:
    torch.Tensor
        The merged depth map in the current camera frame, dimensions [H, W].
    """
    # Transform point clouds to the current camera frame
    pcd_last_cam = torch.einsum("hwj,jk->hwk", pcd_last, torch.inverse(pose_cur))
    pcd_current_cam = torch.einsum("hwj,jk->hwk", pcd_current, torch.inverse(pose_cur))

    # Project points using intrinsics and get depth values
    depth_last = pcd_last_cam[..., 2]
    depth_current = pcd_current_cam[..., 2]

    # Filter points behind the camera by setting their depth to a large value
    large_value = torch.inf  # Can be set to an appropriate large value or torch.inf
    depth_last = torch.where(depth_last > 0, depth_last, large_value)
    depth_current = torch.where(depth_current > 0, depth_current, large_value)

    # Merge depths by selecting the minimum depth at each pixel location
    merged_depth = torch.min(depth_last, depth_current)

    return merged_depth


def _unproject_depth(pcd: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """
    Projects a point cloud from world coordinates back to a depth image using the provided pose.

    Parameters
    ----------
    intrinsics:

    pcd : torch.Tensor
       The point cloud in world coordinates with dimensions [height, width, 4].
    pose : torch.Tensor
       The 4x4 transformation matrix from world to camera coordinates.

    Returns
    -------
    torch.Tensor
       The depth image created from the point cloud with dimensions [height, width].
    """
    # Inverse transform to world coordinates (assuming pcd includes homogeneous coordinate)
    homogenous_world = torch.einsum("hwj,jk->hwk", pcd[..., 2:], torch.inverse(pose))

    # Convert homogeneous coordinates back to Euclidean coordinates in the camera frame
    xyz_camera = homogenous_world[..., :3] / (
        homogenous_world[..., 3:4] + 1e-10
    )  # Adding epsilon to avoid division by zero

    # Project points using intrinsics
    projected = torch.einsum("ij,hwj->hwi", intrinsics, xyz_camera)
    u = projected[:, :, 0] / (projected[:, :, 2] + 1e-10)  # u coordinate
    v = projected[:, :, 1] / (projected[:, :, 2] + 1e-10)  # v coordinate

    # Map (u, v) back to depth values
    # Since depth values are typically in z, directly extract z from the transformed coordinates
    depth_image = xyz_camera[:, :, 2]

    return depth_image


def compute_silhouette_diff(depth: Tensor, rastered_depth: Tensor) -> Tensor:
    """
    Compute the difference between the sobel edges of two depth images.

    Parameters
    ----------
    depth : torch.Tensor
        The depth image with dimensions [height, width].
    rastered_depth : torch.Tensor
        The depth image with dimensions [height, width].

    Returns
    -------
    torch.Tensor
        The silhouette difference between the two depth images with dimensions [height, width].
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    else:
        depth = depth.unsqueeze(1)
    if rastered_depth.dim() == 2:
        rastered_depth = rastered_depth.unsqueeze(0).unsqueeze(0)
    else:
        rastered_depth = rastered_depth.unsqueeze(1)
    edge_depth = kornia.filters.sobel(depth)
    edge_rastered_depth = kornia.filters.sobel(rastered_depth)
    silhouette_diff = torch.abs(edge_depth - edge_rastered_depth).squeeze()
    return silhouette_diff


def normalize_depth(depth):
    min_val = depth.min()
    max_val = depth.max()
    normalized_depth = (depth - min_val) / (max_val - min_val)
    return normalized_depth, min_val, max_val


def unnormalize_depth(normalized_depth, min_val, max_val):
    depth = normalized_depth * (max_val - min_val) + min_val
    return depth


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """
    Convert a quaternion to a rotation matrix.

    Parameters
    ----------
    quaternion : torch.Tensor
        The quaternion with dimensions [4].

    Returns
    -------
    torch.Tensor
        The rotation matrix with dimensions [3, 3].
    """

    normalized_quaternion = quaternion / torch.norm(quaternion)
    return KG.quaternion_to_rotation_matrix(normalized_quaternion)


def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """
    Convert a rotation matrix to a quaternion.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        The rotation matrix with dimensions [3, 3].

    Returns
    -------
    torch.Tensor
        The quaternion with dimensions [4].
    """
    return KG.rotation_matrix_to_quaternion(rotation_matrix)
