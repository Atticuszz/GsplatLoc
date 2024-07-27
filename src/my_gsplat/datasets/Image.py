import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from ..geometry import depth_to_points
from ..utils import DEVICE, remove_outliers, to_tensor


class RGBDImage:
    """
    Initialize an RGBDImage with depth and camera intrinsic matrix, all as Tensors.

    Parameters
    ----------
    depth: np.ndarray
        The depth image as a numpy array.
    K: np.ndarray
        Camera intrinsic matrix as a numpy array.
    depth_scale: float
        Factor by which the depth values are scaled.
    pose: np.ndarray | None, optional
        Camera pose matrix in world coordinates as a numpy array.
    """

    def __init__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float,
        pose: NDArray[np.float32],
    ):
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError(
                "RGB's height and width must match Depth's height and width."
            )
        self._rgb = to_tensor(rgb, device=DEVICE)
        self._depth = to_tensor(depth / depth_scale, device=DEVICE)
        self._K = to_tensor(K, device=DEVICE)
        self._pose = to_tensor(pose, device=DEVICE)
        self._pcd = depth_to_points(self._depth, self._K)

        # NOTE: remove outliers
        self._pcd, inlier_mask = remove_outliers(self._pcd, verbose=True)
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
        pcd : Tensor shape=(h*w, 3)
        """
        return self._pcd

    @points.setter
    def points(self, new_points: Tensor):
        if new_points.shape[1] != 3:
            raise ValueError(
                "Points must be a 2-dimensional tensor with the second dimension of size 3."
            )
        self._pcd = new_points
