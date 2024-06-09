import torch
from torch import Tensor

from src.pose_estimation import DEVICE


def construct_full_pose(rotation, translation):
    """
    Constructs the full 4x4 transformation matrix from rotation and translation.
    Ensures that gradients are tracked for rotation and translation.
    """
    pose = torch.eye(4, dtype=torch.float64, device=DEVICE)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose.requires_grad_(True)  # Ensure that gradients are tracked
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
        The converted point cloud in world coordinates with dimensions [height, width, 4].
    """
    height, width = depth.shape
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width), torch.arange(height), indexing="xy"
    )
    grid_x = grid_x.float().to(DEVICE)
    grid_y = grid_y.float().to(DEVICE)

    Z = depth.to(DEVICE)
    X = (grid_x - intrinsics[0, 2]) * Z / intrinsics[0, 0]
    Y = (grid_y - intrinsics[1, 2]) * Z / intrinsics[1, 1]
    ones = torch.ones_like(Z)

    pcd = torch.stack((X, Y, Z, ones), dim=-1)
    pcd_world = torch.einsum("hwj,jk->hwk", pcd, pose)
    return pcd_world


def unproject_depth(pcd: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """
    Projects a point cloud from world coordinates back to a depth image using the provided pose.

    Parameters
    ----------
    pcd : torch.Tensor
       The point cloud in world coordinates with dimensions [height, width, 4].
    pose : torch.Tensor
       The 4x4 transformation matrix from world to camera coordinates.

    Returns
    -------
    torch.Tensor
       The depth image created from the point cloud with dimensions [height, width].
    """
    pcd_camera = torch.einsum("hwj,jk->hwk", pcd, torch.inverse(pose))

    x = pcd_camera[..., 0]
    y = pcd_camera[..., 1]
    z = pcd_camera[..., 2]
    u = (x / z) * intrinsics[0, 0] + intrinsics[0, 2]
    v = (y / z) * intrinsics[1, 1] + intrinsics[1, 2]

    projected_depth = torch.zeros_like(z)
    height, width = z.shape
    u, v = u.long(), v.long()
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    projected_depth[v[valid], u[valid]] = z[valid]

    return projected_depth
