import numpy as np
from numpy._typing import NDArray


def calculate_translation_error(
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


def calculate_rotation_error(
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


def calculate_pointcloud_rmse(
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


def diff_pcd_COM(pcd_1: NDArray[np.float64], pcd_2: NDArray[np.float64]) -> float:
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
