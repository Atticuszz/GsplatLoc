import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.data.utils import to_tensor
from src.my_gsplat.geometry import depth_to_points
from src.my_gsplat.utils import remove_outliers


class RGBDImage:
    """
    Initialize an RGBDImage with depth and camera intrinsic matrix, all as Tensors.
    """

    def __init__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        pose: NDArray[np.float32],
    ):
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError(
                "RGB's height and width must match Depth's height and width."
            )
        self._rgb = to_tensor(rgb)
        self._depth = to_tensor(depth)
        self._K = to_tensor(K)
        self._pose = to_tensor(pose)
        self._pcd = depth_to_points(self._depth, self._K).view(-1, 3)

        # NOTE: remove outliers
        self._pcd, inlier_mask = remove_outliers(self._pcd, verbose=False)
        self._colors = (self._rgb / 255.0).reshape(-1, 3)[inlier_mask]  # N,3

        # self._colors = (self._rgb / 255.0).reshape(-1, 3)  # N,3

    @property
    def size(self):
        return self._pcd.size(0)

    @property
    def colors(self) -> Tensor:
        """
        normed colors
        Returns
        -------
        colors: Tensor[torch.float32], shape=(n, 3)
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors: Tensor):
        if new_colors.shape[1] != 3:
            raise ValueError(
                "Colors must be a 2-dimensional tensor with the second dimension of size 3."
            )
        if torch.any((new_colors < 0) | (new_colors > 1)):
            raise ValueError("Color values must be in the range [0, 1].")
        self._colors = new_colors

    @property
    def rgbs(self) -> Tensor:
        """
        Returns
        -------
            rgb: Tensor[torch.float32], shape=(h, w, 3)
        """
        return self._rgb

    @property
    def depth(self) -> Tensor:
        """
        Returns
        -------
        depth: Tensor[torch.float32], shape=(h, w)
            Depth image in meters.
        """

        return self._depth

    @depth.setter
    def depth(self, new_depth: Tensor):
        if new_depth.dim() != 2:
            raise ValueError("Depth must be a 2-dimensional matrix.")
        self._depth = new_depth

    @property
    def K(self) -> Tensor:
        """
        Returns
        -------
        K: Tensor[torch.float32], shape=(3, 3)
            Camera intrinsic matrix.
        """
        return self._K

    @K.setter
    def K(self, new_K: Tensor):
        if new_K.shape != (3, 3):
            raise ValueError("Camera intrinsic matrix K must be a 3x3 matrix.")
        self._K = new_K

    @property
    def pose(self) -> Tensor:
        """
        Returns
        -------

        pose: Tensor[torch.float32] | None, shape=(4, 4)
            Camera pose matrix in world coordinates.
        """
        return self._pose

    @pose.setter
    def pose(self, new_pose: Tensor):
        if new_pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix and Tensor.")
        self._pose = new_pose

    @property
    def points(self) -> Tensor:
        """
        in camera pcd
        Returns
        -------
        pcd : Tensor shape=(N, 3)
        """
        return self._pcd

    @points.setter
    def points(self, new_points: Tensor):
        if new_points.shape[1] != 3:
            raise ValueError(
                "Points must be a 2-dimensional tensor with the 2nd dimension of size 3."
            )
        self._pcd = new_points
