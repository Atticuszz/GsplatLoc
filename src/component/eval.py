from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import wandb


class WandbLogger:
    def __init__(self):
        """
        Initialize the Weights & Biases logging.
        """
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="ABGICP",
            entity="supavision",
            config={
                "algorithm": "Improved ICP",
                "dataset": "3D Point Clouds",
                "description": "Testing translation and rotation error metrics",
            },
            name=run_name,
        )

    def log_translation_error(self, eT: float, step: int):
        """
        Log the translation error to wandb.
        """
        wandb.log({"Translation Error": eT}, step=step)

    def log_rotation_error(self, eR: float, step: int):
        """
        Log the rotation error to wandb.
        """
        wandb.log({"Rotation Error": eR}, step=step)

    def log_rmse_pcd(self, rmse: float, step: int):
        """
        Log the point cloud RMSE to wandb.
        """
        wandb.log({"Point Cloud RMSE": rmse}, step=step)

    def log_com_diff(self, com_diff: float, step: int):
        """
        Log the difference in center of mass between two point clouds to wandb.
        """
        wandb.log({"COM Difference": com_diff}, step=step)

    def log_align_time(self, time: float, step: int):
        """
        Log the alignment time to wandb.
        """
        wandb.log({"Alignment Time": time}, step=step)

    def log_downsample_time(self, time: float, step: int):
        """
        Log the downsample time to wandb.
        """
        wandb.log({"Downsample Time": time}, step=step)

    def log_iter_times(self, iter_times: int, step: int):
        """
        Log the iteration times to wandb.
        """
        wandb.log({"Iteration Times": iter_times}, step=step)

    def log_align_error(self, align_error: float, step: int):
        """
        Log the alignment error to wandb.
        """
        wandb.log({"Alignment Error": align_error}, step=step)

    def finish(self):
        """
        Finish the wandb run.
        """
        wandb.finish()


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
    theta = np.arccos((trace_value - 1) / 2)
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
