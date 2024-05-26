import torch
import numpy as np
from torchimize.functions import lsq_lma_parallel
from .pcd import PointClouds


def transform_points(points, transform):
    """Apply geometric transformation to the points."""
    homogeneous_points = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    transformed_points = transform @ homogeneous_points.T
    return transformed_points.T[:, :3]


def compute_geometric_error(source, target, transform, covariances):
    """Compute geometric error using Mahalanobis distance."""
    transformed_source = transform_points(source, transform)
    diff = target - transformed_source
    error = torch.sum(diff.unsqueeze(-2) @ covariances @ diff.unsqueeze(-1))
    return error


def color_error(source_lab, target_lab):
    """Compute color error in L*a*b* space (can be extended to use CIEDE2000)."""
    return torch.norm(source_lab - target_lab, dim=1).sum()


def total_error(
    params,
    source_points,
    target_points,
    source_lab,
    target_lab,
    source_covariances,
    alpha,
    beta,
):
    """Total error combining geometric and color errors."""
    transform = params_to_transform(
        params
    )  # Convert parameters to transformation matrix
    geom_error = compute_geometric_error(
        source_points, target_points, transform, source_covariances
    )
    col_error = color_error(source_lab, target_lab)
    return alpha * geom_error + beta * col_error


# Optimization setup
init_params = torch.zeros(6)  # Parameter vector (e.g., for SE(3) transformation)
alpha = 1.0  # Weight for geometric error
beta = 0.1  # Weight for color error

# Load and preprocess point clouds
source_pcd = PointClouds(source_data, source_rgb)
target_pcd = PointClouds(target_data, target_rgb)
source_pcd.preprocess()
target_pcd.preprocess()


# Define the function for the optimizer
def optimization_func(params):
    return total_error(
        params,
        source_pcd.points,
        target_pcd.points,
        source_pcd.lab,
        target_pcd.lab,
        source_pcd.covs,
        alpha,
        beta,
    )


# Run the optimization
optimized_params = lsq_lma_parallel(init_params, optimization_func, max_iter=100)
