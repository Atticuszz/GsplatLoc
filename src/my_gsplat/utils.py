import json
import math
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import small_gicp
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {DEVICE} DEVICE")


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


class KnnSearch:
    def __init__(self, data: Tensor | NDArray):

        if torch.is_tensor(data):
            pcd = small_gicp.PointCloud(data.detach().cpu().numpy())
            self._kdtree = small_gicp.KdTree(pcd, num_threads=32)
        elif isinstance(data, np.ndarray):
            pcd = small_gicp.PointCloud(data)
            self._kdtree = small_gicp.KdTree(pcd, num_threads=32)
        else:
            raise ValueError("knn search data must be tensor or numpy")

    def query(
        self, target: Tensor | NDArray[np.float64], k: int
    ) -> tuple[Tensor, Tensor] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        knn search
        Parameters
        ----------
        target: Tensor | NDArray ,shape(n,3) or (n,4)
        k: int

        Returns
        -------
            k_indexs: Tensor | NDArray shape(n,k)
            k_sqs: Tensor | NDArray shape(n,k)
        """
        if torch.is_tensor(target):

            k_indexs, k_sqs = self._kdtree.batch_knn_search(
                target.detach().cpu().numpy(), k=k, num_threads=64
            )

            return to_tensor(k_indexs, device=target.device), to_tensor(
                k_sqs, device=target.device
            )
        elif isinstance(target, np.ndarray):

            k_indexs, k_sqs = self._kdtree.batch_knn_search(target, k=k, num_threads=64)
            return np.array(k_indexs, dtype=np.float64), np.array(
                k_sqs, dtype=np.float64
            )
        else:
            raise TypeError("knn search data must be tensor or numpy")


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
    points: torch.Tensor, k: int = 2, std_ratio: float = 5.0, verbose: bool = False
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


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_intrinsics_matrix(intrinsics: list[float]) -> NDArray:
    """
    Get matrix representation of intrinsics.
    :param intrinsics : [fx,fy,cx,cy]
    :return: K matrix.shape=(3,3)
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def load_camera_cfg(cfg_path: str) -> dict:
    """
    Load camera configuration from YAML or Json file.
    """
    cfg_path = Path(cfg_path)
    assert cfg_path.exists(), f"File not found: {cfg_path}"
    with open(cfg_path) as file:
        if cfg_path.suffix in [".yaml", ".yml"]:
            cfg = yaml.safe_load(file)
        elif cfg_path.suffix == ".json":
            cfg = json.load(file)
        else:
            raise TypeError(f"Failed to load cfg via:{cfg_path.suffix}")
    return cfg


def to_tensor(data, device=DEVICE, requires_grad=False, dtype=torch.float32):
    """
    Convert numpy array or list to a PyTorch tensor.
    """
    if (
        not isinstance(data, list)
        and not isinstance(data, np.ndarray)
        and not torch.is_tensor(data)
        and not isinstance(data, int)
    ):
        raise TypeError("to tensor needs list,np.ndarray or tensor")

    data = torch.as_tensor(data, dtype=dtype, device=device)  # More efficient
    return data.requires_grad_(requires_grad)


def calculate_translation_error(
    estimated_pose: torch.Tensor, true_pose: torch.Tensor
) -> float:
    """
    Calculate the translation error between estimated pose and true pose using PyTorch.
    Parameters
    ----------
    estimated_pose: torch.Tensor, shape=(4, 4)
    true_pose: torch.Tensor, shape=(4, 4)

    Returns
    -------
    translation_error: float
    """
    t_est = estimated_pose[:3, 3]
    t_true = true_pose[:3, 3]
    translation_error = torch.norm(t_est - t_true).item()
    return translation_error


def calculate_rotation_error(
    estimated_pose: torch.Tensor, true_pose: torch.Tensor
) -> float:
    """
    Calculate the rotation error between estimated pose and true pose using PyTorch.
    Parameters
    ----------
    estimated_pose: torch.Tensor, shape=(4, 4)
    true_pose: torch.Tensor, shape=(4, 4)

    Returns
    -------
    rotation_error: float
    """
    R_est = estimated_pose[:3, :3]
    R_true = true_pose[:3, :3]
    delta_R = torch.mm(R_est, R_true.transpose(0, 1))
    trace_value = torch.trace(delta_R)
    cos_theta = (trace_value - 1) / 2
    cos_theta = torch.clamp(
        cos_theta, -1, 1
    )  # # Ensure the value is within a valid range
    theta = torch.acos(cos_theta)

    # Convert radians to degrees manually
    rotation_error = (theta * 180 / torch.pi).item()
    return rotation_error


def calculate_ate(
    estimated_poses: list[torch.Tensor], true_poses: list[torch.Tensor]
) -> float:
    """
    Calculate the Average Trajectory Error (ATE) for a sequence of poses.
    ATE is defined as the root mean square of the translation errors.

    Parameters
    ----------
    estimated_poses: list of torch.Tensor, each with shape=(4, 4)
        List of estimated pose matrices
    true_poses: list of torch.Tensor, each with shape=(4, 4)
        List of true pose matrices

    Returns
    -------
    ate: float
        Average Trajectory Error (root mean square of translation errors)
    """
    squared_errors = []
    for est_pose, true_pose in zip(estimated_poses, true_poses):
        error = calculate_translation_error(est_pose, true_pose)
        squared_errors.append(error**2)

    mean_squared_error = sum(squared_errors) / len(squared_errors)
    ate = math.sqrt(mean_squared_error)
    return ate


def calculate_rre(
    estimated_poses: list[torch.Tensor], true_poses: list[torch.Tensor]
) -> float:
    """
    Calculate the Root mean square Rotation Error (RRE) for a sequence of poses.

    Parameters
    ----------
    estimated_poses: list of torch.Tensor, each with shape=(4, 4)
        List of estimated pose matrices
    true_poses: list of torch.Tensor, each with shape=(4, 4)
        List of true pose matrices

    Returns
    -------
    rre: float
        Root mean square Rotation Error (in degrees)
    """
    squared_errors = []
    for est_pose, true_pose in zip(estimated_poses, true_poses):
        error = calculate_rotation_error(est_pose, true_pose)
        error_rad = error * math.pi / 180  # Convert back to radians
        squared_errors.append(error_rad**2)

    mean_squared_error = sum(squared_errors) / len(squared_errors)
    rre = math.sqrt(mean_squared_error)
    return rre * 180 / math.pi  # Convert final result back to degrees


def visualize_point_cloud(points: torch.Tensor, colors: torch.Tensor):
    """
    使用Open3D可视化点云数据。

    参数:
    points (torch.Tensor): 形状为(N, 3)的点坐标张量。
    colors (torch.Tensor): 形状为(N, 3)的颜色张量，值范围在0到1之间。

    """
    # 确保输入是正确的形状
    assert points.shape[1] == 3, "点坐标应该是(N, 3)形状"
    assert colors.shape[1] == 3, "颜色应该是(N, 3)形状"
    assert points.shape[0] == colors.shape[0], "点和颜色的数量应该相同"

    # 转换为numpy数组
    points_np = points.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
