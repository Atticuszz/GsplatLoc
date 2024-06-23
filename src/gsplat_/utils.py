import random

import numpy as np
import small_gicp
import torch
from numpy._typing import NDArray
from torch import Tensor

from src.pose_estimation import DEVICE
from src.utils import to_tensor


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
    x_np = x.cpu().numpy()
    pcd = small_gicp.PointCloud(x_np)
    model = small_gicp.KdTree(pcd, num_threads=32)
    _, distances = model.batch_knn_search(x_np, k=K, num_threads=64)
    return to_tensor(distances, device=DEVICE, requires_grad=True)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
