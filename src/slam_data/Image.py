import numpy as np
import torch
from numpy.compat import long
from numpy.typing import NDArray
from torch import Tensor

from src.pose_estimation import DEVICE
from src.utils import to_tensor


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
        pose: NDArray[np.float64],
    ):
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError(
                "RGB's height and width must match Depth's height and width."
            )
        self._rgb = to_tensor(rgb, device=DEVICE, requires_grad=True)
        self._depth = to_tensor(depth / depth_scale, device=DEVICE, requires_grad=True)
        self._K = to_tensor(K, device=DEVICE, requires_grad=True)

        self._pose = to_tensor(pose, device=DEVICE, requires_grad=True)
        self._pcd = self._project_pcds(include_homogeneous=False)

    @property
    def color(self) -> Tensor:
        """
        Returns
        -------
        color: Tensor[torch.float64], shape=(h, w, 3)
        """
        return self._rgb

    @property
    def depth(self) -> Tensor:
        """
        Returns
        -------
        depth: Tensor[torch.float64], shape=(h, w)
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
        K: Tensor[torch.float64], shape=(3, 3)
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

        pose: Tensor[torch.float64] | None, shape=(4, 4)
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

    def _project_pcds(
        self,
        include_homogeneous: bool = True,
    ) -> Tensor:
        """
        Project depth map to point clouds using intrinsic matrix.

        Parameters
        ----------
        include_homogeneous: bool, optional
            Whether to include the homogeneous coordinate (default True).

        Returns
        -------
        points: Tensor
            The generated point cloud, shape=(h*w, 3) or (h*w, 4).
        """
        h, w = self.depth.shape
        i_indices, j_indices = torch.meshgrid(
            torch.arange(h, device=DEVICE),
            torch.arange(w, device=DEVICE),
            indexing="ij",
        )
        x = (j_indices - self.K[0, 2]) * self.depth / self.K[0, 0]
        y = (i_indices - self.K[1, 2]) * self.depth / self.K[1, 1]
        z = self.depth

        points = torch.stack((x, y, z), dim=-1)  # shape (h, w, 3)

        if include_homogeneous:
            ones = torch.ones((h, w, 1), device=DEVICE)
            points = torch.cat((points, ones), dim=-1)  # shape (h, w, 4)

        # Flatten the points to shape (h*w, 3) or (h*w, 4)
        points = points.reshape(-1, points.shape[-1])

        return points

    def _color_pcds(
        self,
        colored: bool = False,  # Optional color image
        include_homogeneous: bool = True,
    ) -> NDArray[np.float64]:
        """
        Generate point clouds from depth image, optionally with color.

        Parameters
        ----------
        colored: bool
            if contain color
        include_homogeneous : bool, optional
            Whether to include homogeneous coordinate.

        Returns
        -------
        NDArray[np.float64]
            The generated point cloud, shape=(h*w, 4) or (h*w, 6) or (h*w, 3) or (h*w, 7) depending on options.
        """
        h, w = self._depth.shape[:2]
        i_indices, j_indices = np.indices((h, w))

        # Transform to camera coordinates using intrinsic matrix K
        x = (j_indices - self._K[0, 2]) * self._depth / self._K[0, 0]
        y = (i_indices - self._K[1, 2]) * self._depth / self._K[1, 1]
        z = self._depth
        points = np.stack((x, y, z), axis=-1)

        # Handle the optional inclusion of the homogeneous coordinate
        if include_homogeneous:
            ones = np.ones((h, w, 1))
            points = np.concatenate((points, ones), axis=-1)

        # Flatten the array to make it N x (3 or 4)
        points = points.reshape(-1, points.shape[-1])

        # Handle the optional color image
        if colored:
            colors = self._rgb.reshape(-1, 3)  # Flatten the color array
            # Stack the color coordinates with the points
            points = np.concatenate((points, colors), axis=1)

        return points

    def _pointclouds(
        self, stride: int = 1, include_homogeneous=True
    ) -> NDArray[np.float64]:
        """
        Generate point clouds from depth image.
        Parameters
        ----------
        stride: int, optional
        include_homogeneous: bool, optional, whether to include homogeneous coordinate
        Returns
        -------
        pcd: NDArray[np.float64], shape=(h*w, 4) or (h*w, 3)
        """
        i_indices, j_indices, depth_downsampled = self._grid_downsample(stride)
        # Transform to camera coordinates
        x = (j_indices - self._K[0, 2]) * depth_downsampled / self.K[0, 0]
        y = (i_indices - self._K[1, 2]) * depth_downsampled / self.K[1, 1]
        z = depth_downsampled

        points = np.stack((x, y, z), axis=-1)

        # Add homogeneous coordinate if requested
        if include_homogeneous:
            ones = np.ones((points.shape[0], points.shape[1], 1))
            points_homogeneous = np.concatenate((points, ones), axis=-1)
            return points_homogeneous.reshape(-1, 4)
        else:
            return points.reshape(-1, 3)

    def _camera_to_world(
        self,
        c2w: np.ndarray,
        pcd_c: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Transform points from camera coordinates to world coordinates using the c2w matrix.
        :param c2w: 4x4 transformation matrix from camera to world coordinates
        :param pcd_c: Nx4 numpy array of 3D points in camera coordinates
        :return: Nx4 numpy array of transformed 3D points in world coordinates
        """
        if pcd_c is None:
            points_camera = self._pointclouds()
        else:
            points_camera = pcd_c[:, :4]
        return points_camera @ c2w.T

    def _grid_downsample(self, stride: int = 1) -> tuple[
        NDArray[np.signedinteger | long],
        NDArray[np.signedinteger | long],
        NDArray[np.float64],
    ]:
        """
        Parameters
        ----------
        stride: int, optional
        Returns
        -------
        i_indices: NDArray[np.signedinteger | long], shape=(h, w)
            Pixel indices along the height axis.
        j_indices: NDArray[np.signedinteger | long], shape=(h, w)
            Pixel indices along the width axis.
        depth_downsampled: NDArray[np.float64], shape=(h, w)
            Downsampled depth image.
        """
        # Generate pixel indices
        i_indices, j_indices = np.indices(self.depth.shape)
        # Apply downsampling
        return (
            i_indices[::stride, ::stride],
            j_indices[::stride, ::stride],
            self._depth[::stride, ::stride],
        )
