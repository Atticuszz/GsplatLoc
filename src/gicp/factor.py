import numpy as np
import torch
from numpy._typing import NDArray

from .pcd import PointClouds
from ..utils import to_tensor


def gicp_error_optimized(
    point_clouds_target: PointClouds, point_clouds_source: PointClouds, T: torch.Tensor
):
    """
    Compute the GICP error using nearest neighbor search with a k-d tree.

    :param point_clouds_target: Target point clouds object
    :param point_clouds_source: Source point clouds object
    :param T: Transformation matrix [4, 4]
    :return: Total error as a single float
    """
    # Transform the source points
    pcd_source = to_tensor(point_clouds_source.points)
    transformed_source = torch.matmul(T, pcd_source.T).T[:, :3]
    pcd_target = to_tensor(point_clouds_target.points)
    # Use k-d tree to find nearest neighbors in the target
    _, indices = point_clouds_target.kdtree.query(transformed_source.numpy(), k=1)
    nearest_target_points = pcd_target[indices.flatten()]

    # Compute residuals
    residuals = nearest_target_points - transformed_source

    # Calculate the Mahalanobis distance for each matched pair
    total_error = 0.0
    for i, idx in enumerate(indices.flatten()):
        cov_source = point_clouds_source.cov(i)
        cov_target = point_clouds_target.cov(idx)
        RCR = cov_target + torch.matmul(
            torch.matmul(T[:3, :3], cov_source[:3, :3]), T[:3, :3].T
        )
        mahalanobis_weight = torch.inverse(RCR)
        error = torch.matmul(
            torch.matmul(residuals[i].unsqueeze(0), mahalanobis_weight),
            residuals[i].unsqueeze(1),
        )
        total_error += 0.5 * error

    return total_error.squeeze()


def gicp_single_point_error(
    point_clouds_target: PointClouds,
    point_source: NDArray[np.float64],
    T: NDArray[np.float64],
):
    """
    Compute the GICP error for a single source point against a target point cloud using k-d tree.
    Assumes source and target points include homogeneous coordinates [x, y, z, 1].

    :param point_clouds_target: Target point clouds object
    :param point_source: Source point as a tensor [4] (homogeneous coordinates)
    :param T: Transformation matrix [4, 4]
    :return: Total error as a single float, Index of the nearest target point
    """
    # Transform the source point using the full 4x4 transformation matrix
    transformed_source = T @ point_source

    # Use k-d tree to find the nearest neighbor in the target
    found, index, sq_dist = point_clouds_target.kdtree.nearest_neighbor_search(
        transformed_source
    )
    if not found:
        return False

    nearest_target_point = point_clouds_target.point(index)

    # Compute residual for the first 3 dimensions (x, y, z)
    residual = nearest_target_point[:3] - transformed_source[:3]
    assert np.allclose(residual, sq_dist)

    # Retrieve the covariance of the nearest target point
    cov_target = point_clouds_target.cov(index)

    # TODO: finish loss func
    # Calculate the combined covariance matrix for the Mahalanobis distance
    RCR = cov_target + torch.matmul(torch.matmul(T[:3, :3], cov_source), T[:3, :3].T)
    mahalanobis_weight = torch.inverse(RCR)

    # Calculate the Mahalanobis distance error
    error = (
        0.5
        * (residual.unsqueeze(0) @ mahalanobis_weight @ residual.unsqueeze(1)).item()
    )

    return error, nearest_target_index
