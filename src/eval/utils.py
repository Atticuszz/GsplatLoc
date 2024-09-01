import json
import pathlib
import random
import shutil

import kornia
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor


def calculate_translation_error_np(
    estimated_pose: NDArray[np.float64], true_pose: NDArray[np.float64]
) -> float:
    """
    Calculate the translation error between estimated pose and true pose.
    Parameters
    ----------
    estimated_pose: NDArray[np.float64], shape=(4, 4)
    true_pose: NDArray[np.float64], shape=(4, 4)

    Returns
    -------
    translation_error: float
    """
    # 提取平移向量
    t_est = estimated_pose[:3, 3]
    t_true = true_pose[:3, 3]
    # 计算欧氏距离
    translation_error = np.linalg.norm(t_est - t_true)
    return translation_error


def calculate_rotation_error_np(
    estimated_pose: NDArray[np.float64], true_pose: NDArray[np.float64]
) -> float:
    """
    Calculate the rotation error between estimated pose and true pose.
    Parameters
    ----------
    estimated_pose: NDArray[np.float64], shape=(4, 4)
    true_pose: NDArray[np.float64], shape=(4, 4)

    Returns
    -------
    rotation_error: float
    """
    # 提取旋转矩阵
    R_est = estimated_pose[:3, :3]
    R_true = true_pose[:3, :3]
    # 计算相对旋转矩阵
    delta_R = R_est @ R_true.T
    # 计算旋转角度
    trace_value = np.trace(delta_R)
    cos_theta = (trace_value - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # 确保值在合法范围内
    theta = np.arccos(cos_theta)

    # 返回以度为单位的旋转误差
    rotation_error = np.degrees(theta)
    return rotation_error


def calculate_pointcloud_rmse_np(
    estimated_points: NDArray[np.float64], true_points: NDArray[np.float64]
) -> float:
    """
    Calculate the RMSE between estimated points and true points.
    Parameters
    ----------
    estimated_points: NDArray[np.float64], shape=(n, 3) or (n, 4)
    true_points: NDArray[np.float64], shape=(n, 3) or (n, 4)

    Returns
    -------
    rmse: float
    """
    if estimated_points.shape[1] == 4:
        estimated_points = estimated_points[:, :3]
    if true_points.shape[1] == 4:
        true_points = true_points[:, :3]
    differences = estimated_points - true_points
    squared_differences = np.sum(differences**2, axis=1)
    rmse = np.sqrt(np.mean(squared_differences))
    return rmse


def diff_pcd_COM_np(pcd_1: NDArray[np.float64], pcd_2: NDArray[np.float64]) -> float:
    """
    Calculate the difference in center of mass between two
    point clouds.
    Parameters
    ----------
    pcd_1: NDArray[np.float64], shape=(n, 3)
    pcd_2: NDArray[np.float64], shape=(n, 3)

    Returns
    -------
    diff_COM: NDArray[np.float64], shape=(3,)
    """
    if pcd_1.shape[1] == 4:
        pcd_1 = pcd_1[:, :3]
    if pcd_2.shape[1] == 4:
        pcd_2 = pcd_2[:, :3]
    com1 = np.mean(pcd_1, axis=0)
    com2 = np.mean(pcd_2, axis=0)
    distance = np.linalg.norm(com1 - com2)
    return distance


def calculate_RMSE_np(eT: NDArray) -> float:
    """
    Returns
    -------
    RMSE: float
    """
    return np.sqrt(np.mean(np.square(eT)))


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


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def count_images(media_dir):
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    return sum(
        1 for file in media_dir.glob("**/*") if file.suffix.lower() in image_extensions
    )


def clean_wandb_runs(wandb_dir):
    wandb_path = pathlib.Path(wandb_dir)

    for run_dir in wandb_path.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "files" / "config.yaml"
        summary_path = run_dir / "files" / "wandb-summary.json"
        media_dir = run_dir / "files" / "media"
        # 检查 config.yaml 是否存在并包含正确的 dataset 值
        if config_path.exists():
            with open(config_path) as config_file:
                config = yaml.safe_load(config_file)
                dataset_value = config.get("dataset", {}).get("value")
                if dataset_value != "Replica":
                    continue
        else:
            continue

        # 检查 wandb-summary.json 中的 _step 值
        if summary_path.exists():
            with open(summary_path) as summary_file:
                summary = json.load(summary_file)
                step_value = summary.get("_step")
                if step_value is None or step_value >= 1900:
                    continue
        else:
            continue

        # 检查并打印 media 目录下图片数量
        if media_dir.exists():
            image_count = count_images(media_dir)
            if image_count > 1900:
                print(f"Run with more than 1900 images: {run_dir}")
                print(f"Image count: {image_count}")
                continue
        print(f"Removing run directory: {run_dir}")
        shutil.rmtree(run_dir)
