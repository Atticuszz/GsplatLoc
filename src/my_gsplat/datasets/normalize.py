import torch
from torch import Tensor

from .Image import RGBDImage


@torch.no_grad()
def align_principle_axes(point_cloud: torch.Tensor) -> torch.Tensor:
    """
    Align the principal axes of a point cloud to the coordinate axes using PCA.

    Parameters
    ----------
    point_cloud : torch.Tensor
        Nx3 tensor containing the 3D point cloud.

    Returns
    -------
    torch.Tensor
        A 4x4 transformation matrix that aligns the point cloud along principal axes.
    """
    # Compute centroid
    centroid = torch.median(point_cloud, dim=0).values

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = torch.cov(translated_point_cloud.t())

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = eigenvalues.argsort(descending=True)
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant is negative, flip an eigenvector.
    if torch.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.t()

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = torch.eye(4, device=point_cloud.device)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -torch.mv(rotation_matrix, centroid)

    return transform


@torch.no_grad()
def transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Transform points using a SE(3) transformation matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        A 4x4 SE(3) transformation matrix.
    points : torch.Tensor
        An Nx3 tensor of points to be transformed.

    Returns
    -------
    torch.Tensor
        An Nx3 tensor of transformed points.
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return torch.addmm(matrix[:3, 3], points, matrix[:3, :3].t())


@torch.no_grad()
def transform_cameras(
    matrix: torch.Tensor, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a SE(3) transformation to a set of camera-to-world matrices.

    Parameters
    ----------
    matrix : torch.Tensor
        A 4x4 SE(3) transformation matrix.
    c2w : torch.Tensor
        An Nx4x4 tensor of camera-to-world matrices.

    Returns
    -------
    torch.Tensor
        An Nx4x4 tensor of transformed camera-to-world matrices.
    """
    assert matrix.shape == (4, 4)
    assert len(c2w.shape) == 3 and c2w.shape[1:] == (4, 4)
    # Perform the matrix multiplication with einsum for better control
    transformed = torch.einsum("ki,nij->nkj", matrix, c2w)

    # Normalize the 3x3 rotation matrices to maintain scale: Use the norm of the first row
    scaling = torch.norm(transformed[:, 0, :3], p=2, dim=1, keepdim=True)
    transformed[:, :3, :3] /= scaling.unsqueeze(
        -1
    )  # Unsqueeze to match the shape for broadcasting
    return transformed, scaling


@torch.no_grad()
def normalize_2C(tar: RGBDImage, src: RGBDImage) -> tuple[RGBDImage, RGBDImage, Tensor]:
    # calculate tar points normalization transform
    points = tar.points
    transform = align_principle_axes(points)

    # apply transform
    scale_factor = apply_normalize_T(tar, transform)
    scale_factor = apply_normalize_T(src, transform)

    return tar, src, scale_factor


def apply_normalize_T(tar: RGBDImage, T: Tensor) -> Tensor:
    # NOTE: must in world
    tar.points = transform_points(T, tar.points)
    normed_tar_pose, scale_factor = transform_cameras(T, tar.pose.unsqueeze(0))
    tar.pose = normed_tar_pose.squeeze(0)
    return scale_factor
