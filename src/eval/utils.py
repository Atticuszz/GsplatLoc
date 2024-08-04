import random

import numpy as np
import torch
from numpy.typing import NDArray


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
