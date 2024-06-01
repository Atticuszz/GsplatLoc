import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .pcd import PointClouds
from ..utils import to_tensor


def compute_geometric_residuals(
    target: NDArray[np.float64] | Tensor,
    trans_src: NDArray[np.float64] | Tensor,
    T: NDArray[np.float64] | Tensor,
    cov_target: NDArray[np.float64] | Tensor,
    cov_src: NDArray[np.float64] | Tensor,
) -> float | Tensor:

    residual = target[:3] - trans_src[:3]
    # inf error init
    error = float("inf")
    if isinstance(target, np.ndarray):
        # Rotation part of the transformation matrix
        combined_covariance = cov_target + np.dot(np.dot(T, cov_src), T.T)
        inv_combined_covariance = np.linalg.inv(combined_covariance[:3, :3])

        # Calculate Mahalanobis distance
        error = 0.5 * np.dot(residual.T, np.dot(inv_combined_covariance, residual))
        return error
    elif torch.is_tensor(target):
        combined_covariance = cov_target + torch.matmul(
            torch.matmul(T, cov_src), T.transpose(0, 1)
        )
        inv_combined_covariance = torch.inverse(combined_covariance[:3, :3])

        # Calculate Mahalanobis distance
        error = 0.5 * torch.matmul(
            torch.matmul(residual.unsqueeze(0), inv_combined_covariance),
            residual.unsqueeze(1),
        )
        return error


# NOTE: pure math version
def gicp_error_total(T: torch.Tensor, pcd_target: PointClouds, pcd_src: PointClouds):
    """
    Compute the total GICP error for all points in a source point cloud against a target point cloud
    using tensors for all computations.

    Parameters:
    ----------
    pcd_target : Target point clouds object
    pcd_src : Source point clouds object
    T : Transformation matrix [4, 4] as a PyTorch tensor

    Returns:
    ----------
    total_error : float, sum of computed Mahalanobis distances for all points
    """
    # Transform the entire source point cloud using the transformation matrix T
    transformed_sources = torch.matmul(
        T, to_tensor(pcd_src.points).T
    ).T  # Apply transformation

    # Initialize total error
    total_error = []

    # Iterate over each transformed source point
    for idx in range(transformed_sources.shape[0]):
        transformed_source = transformed_sources[idx]

        # Assuming a method to find the nearest neighbor and its index
        found, nearest_idx, _ = pcd_target.kdtree.nearest_neighbor_search(
            transformed_source.detach().cpu().numpy()[:3]
        )
        if not found:
            continue  # If no nearest neighbor is found, skip this point

        nearest_target_point = to_tensor(pcd_target.point(nearest_idx))
        cov_target = to_tensor(pcd_target.cov(nearest_idx))
        cov_source = to_tensor(pcd_src.cov(idx))

        # Calculate the Mahalanobis distance for the matched pair using provided function
        error = compute_geometric_residuals(
            target=nearest_target_point,
            trans_src=transformed_source,
            T=T,
            cov_target=cov_target,
            cov_src=cov_source,
        )

        total_error.append(error)

    return to_tensor(total_error)


# NOTE: small_gicp version
def gicp_error_np(
    pcd_target: PointClouds,
    pcd_src: PointClouds,
    src_idx: int,
    T: NDArray[np.float64],
):
    """
    Compute the GICP error for a single source point against a target point cloud using k-d tree.
    Assumes source and target points include homogeneous coordinates [x, y, z, 1].
    Parameters
    ----------
        pcd_target: Target point clouds object
        pcd_src: Source point clouds object
        src_idx: Index of the source point
        T: Transformation matrix [4, 4]
    Returns
    ----------
        error: float, computed Mahalanobis distance or -1 if no neighbor found
        index: int, index of the nearest neighbor or -1 if no neighbor found
    """
    # Transform the source point using the full 4x4 transformation matrix
    transformed_source = T @ pcd_src.point(src_idx)

    # Use k-d tree to find the nearest neighbor in the target
    found, index, sq_dist = pcd_target.kdtree.nearest_neighbor_search(
        transformed_source
    )
    if not found:
        return False

    nearest_target_point = pcd_target.point(index)

    # Compute residual for the first 3 dimensions (x, y, z)
    residual = nearest_target_point[:3] - transformed_source[:3]
    assert np.allclose(residual, sq_dist), "Residuals do not match!"

    error = compute_geometric_residuals(
        nearest_target_point,
        transformed_source,
        T,
        pcd_target.cov(index),
        pcd_src.cov(src_idx),
    )

    return error, index


def gicp_error_tensor(
    pcd_target: PointClouds,
    pcd_src: PointClouds,
    src_idx: int,
    T: Tensor,
):
    """
    Compute the GICP error for a single source point against a target point cloud using k-d tree.
    Assumes source and target points include homogeneous coordinates [x, y, z, 1].

    Parameters:
    ----------
    pcd_target : Target point clouds object
    pcd_src : Source point clouds object
    src_idx : Index of the source point
    T : Transformation matrix [4, 4]

    Returns:
    ----------
    error : float, computed Mahalanobis distance or -1 if no neighbor found
    index : int, index of the nearest neighbor or -1 if no neighbor found
    """
    src_point = to_tensor(pcd_src.point(src_idx))
    transformed_source = torch.matmul(T, src_point)

    # Assuming the nearest neighbor search is done on the CPU and results are obtained
    found, index, sq_dist = pcd_target.kdtree.nearest_neighbor_search(
        transformed_source.cpu().numpy()
    )
    if not found:
        return -1, -1

    nearest_target_point = to_tensor(pcd_target.point(index))
    residual = nearest_target_point[:3] - transformed_source[:3]
    assert torch.allclose(residual, to_tensor(sq_dist)), "Residuals do not match!"

    error = compute_geometric_residuals(
        nearest_target_point,
        transformed_source,
        T,
        to_tensor(pcd_target.cov(index)),
        to_tensor(pcd_src.cov(src_idx)),
    )

    return error, index
