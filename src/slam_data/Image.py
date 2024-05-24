import numpy as np
from numpy.compat import long
from numpy.typing import NDArray


class RGBDImage:
    def __init__(
        self,
        # rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float,
        pose: NDArray[np.float64] | None = None,
    ):
        # if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
        #     raise ValueError(
        #         "RGB's height and width must match Depth's height and width."
        #     )
        # self.rgb = rgb
        self._depth = depth / depth_scale
        self._K = K
        self._pose: NDArray[np.float64] | None = pose

    @property
    def depth(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        depth: NDArray[np.float64], shape=(h, w)
            Depth image in meters.
        """

        return self._depth

    @property
    def K(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        K: NDArray[np.float64], shape=(3, 3)
            Camera intrinsic matrix.
        """
        return self._K

    @property
    def pose(self) -> NDArray[np.float64] | None:
        """
        Returns
        -------
        pose: NDArray[np.float64] | None, shape=(4, 4)
            Camera pose matrix in world coordinates.
        """
        return self._pose

    def pointclouds(
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

    def camera_to_world(
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
            points_camera = self.pointclouds()
        else:
            points_camera = pcd_c
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
