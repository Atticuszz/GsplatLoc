import torch
from torch import Tensor

from .Image import RGBDImage
from .base import TrainData


@torch.no_grad()
@torch.compile
def similarity_from_cameras(
    c2w: torch.Tensor, strict_scaling: bool = False, center_method: str = "focus"
) -> torch.Tensor:
    """
    Calculate a similarity transformation that aligns and scales camera positions.

    Parameters
    ----------
    c2w : torch.Tensor
        A batch of camera-to-world transformation matrices of shape (N, 4, 4).
    strict_scaling : bool, optional
        If True, use the maximum distance for scaling, otherwise use the median.
    center_method : str, optional
        Method for centering the scene, either "focus" for focusing method or "poses" for camera poses centering.

    Returns
    -------
    torch.Tensor
        A 4x4 similarity transformation matrix that aligns, centers, and scales the input cameras.

    Raises
    ------
    ValueError
        If the `center_method` is not recognized.
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # Rotate the world so that z+ is the up axis
    ups = torch.sum(R * torch.tensor([0, -1.0, 0], device=R.device), dim=-1)
    world_up = torch.mean(ups, dim=0)
    world_up /= torch.norm(world_up)

    up_camspace = torch.tensor([0.0, -1.0, 0.0], device=R.device)
    c = torch.dot(up_camspace, world_up)
    cross = torch.linalg.cross(world_up, up_camspace)
    skew = torch.tensor(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        device=R.device,
    )

    if c > -1:
        R_align = torch.eye(3, device=R.device) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=R.device
        )

    R = R_align @ R
    fwds = torch.sum(R * torch.tensor([0, 0.0, 1.0], device=R.device), dim=-1)
    t = (R_align @ t.unsqueeze(-1)).squeeze(-1)

    # Recenter the scene
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(dim=-1).unsqueeze(-1) * fwds
        translate = -torch.median(nearest, dim=0)[0]
    elif center_method == "poses":
        translate = -torch.median(t, dim=0)[0]
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = torch.eye(4, device=R.device)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # Rescale the scene using camera distances
    scale_fn = torch.max if strict_scaling else torch.median
    scale = 1.0 / scale_fn(torch.norm(t + translate, dim=-1))
    transform[:3, :] *= scale

    return transform


@torch.no_grad()
@torch.compile
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
@torch.compile
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
@torch.compile
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
def normalize_dataset_slice(dataset_slice: list[RGBDImage]) -> list[RGBDImage]:
    all_poses = [rgb_d.pose for rgb_d in dataset_slice]
    # NOTE: transform to world,init with first pose
    all_points = [transform_points(rgb_d.pose, rgb_d.points) for rgb_d in dataset_slice]
    # all_points = [transform_points(all_poses[0], rgb_d.points) for rgb_d in dataset_slice]
    # combine as one scene
    poses = torch.stack(all_poses, dim=0)
    points = torch.cat(all_points, dim=0)

    # normalize
    T1 = similarity_from_cameras(poses)
    poses = transform_cameras(T1, poses)
    points = transform_points(T1, points)

    T2 = align_principle_axes(points)
    poses = transform_cameras(T2, poses)
    points = transform_points(T2, points)

    # transform = T2 @ T1

    # Update the original data with normalized values
    start_idx = 0
    for i, rgb_d in enumerate(dataset_slice):
        num_points = len(rgb_d.points)
        rgb_d.pose = poses[i]
        rgb_d.points = points[start_idx : start_idx + num_points]
        start_idx += num_points

    return dataset_slice


@torch.compile
@torch.no_grad()
def normalize_2C(tar: RGBDImage, src: RGBDImage) -> tuple[RGBDImage, RGBDImage, Tensor]:
    """normalize two rgb-d image with tar.pose"""
    pose = tar.pose.unsqueeze(0)  # -> N,4,4
    # calculate tar points normalization transform
    points = tar.points
    T1 = similarity_from_cameras(pose)
    T2 = align_principle_axes(transform_points(T1, points))
    transform = T2 @ T1

    # apply transform
    tar.points = transform_points(transform, tar.points)
    src.points = transform_points(transform, src.points)
    normed_tar_pose, _ = transform_cameras(transform, tar.pose.unsqueeze(0))
    tar.pose = normed_tar_pose.squeeze(0)
    normed_src_pose, scale_factor = transform_cameras(transform, src.pose.unsqueeze(0))
    src.pose = normed_src_pose.squeeze(0)
    return tar, src, scale_factor


def apply_normalize_T(tar: TrainData, T: Tensor) -> None:
    # NOTE: must in world
    tar.points = transform_points(T, tar.points)
    normed_tar_pose, scale_factor = transform_cameras(T, tar.c2w.unsqueeze(0))
    tar.pose = normed_tar_pose.squeeze(0)
    tar.scale_factor = scale_factor


def normalize_T(tar: RGBDImage) -> Tensor:
    """normalize rgb-d image with PCA"""
    pose = tar.pose.unsqueeze(0)  # -> N,4,4 pose in world
    # calculate tar points normalization transform
    points = tar.points
    T1 = similarity_from_cameras(pose)
    T2 = align_principle_axes(transform_points(T1, points))
    transform = T2 @ T1

    return transform


@torch.no_grad()
def scene_scale(dataset_slice: list[RGBDImage], global_scale: float = 1.0) -> float:
    poses = torch.stack([rgb_d.pose for rgb_d in dataset_slice], dim=0)

    camera_locations = poses[:, :3, 3]
    # assert len(camera_locations) == 2, "Exactly two camera locations are required"

    scene_center = torch.mean(camera_locations, dim=0)
    dists = torch.norm(camera_locations - scene_center, dim=1)
    scale = torch.max(dists)

    return scale.item() * 1.1 * global_scale
