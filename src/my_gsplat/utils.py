import numpy as np
import small_gicp
import torch
from torch import Tensor

from src.data.base import DEVICE
from src.data.utils import to_tensor

# def knn(x: Tensor, K: int = 4) -> Tensor:
#     x_np = x.cpu().numpy()
# model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
# distances, _ = model.kneighbors(x_np)
#     return torch.from_numpy(distances).to(x)


def knn(x: Tensor, K: int = 4) -> Tensor:

    x_np = x.cpu().numpy() if not x.requires_grad else x.detach().cpu().numpy()
    pcd = small_gicp.PointCloud(x_np.astype(np.float64))
    model = small_gicp.KdTree(pcd, num_threads=32)
    _, distances = model.batch_knn_search(x_np, k=K, num_threads=64)
    return to_tensor(distances, device=DEVICE, requires_grad=True)


def remove_outliers(
    points: torch.Tensor, k: int = 10, std_ratio: float = 10.0, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate distances for initial scale and outlier detection
    distances = knn(points, k)
    dist2_avg = (distances[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)

    # Outlier detection
    mean_dist = dist_avg.mean()
    std_dist = dist_avg.std()
    threshold = mean_dist + std_ratio * std_dist
    inlier_mask = dist_avg < threshold

    # Remove outliers
    cleaned_points = points[inlier_mask]

    if verbose:
        num_original = len(points)
        num_cleaned = len(cleaned_points)
        percent_removed = (num_original - num_cleaned) / num_original * 100
        print(f"Original points: {num_original}")
        print(f"Points after cleaning: {num_cleaned}")
        print(f"Percentage removed: {percent_removed:.2f}%")

    return cleaned_points, inlier_mask


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
