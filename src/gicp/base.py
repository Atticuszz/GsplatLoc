import numpy as np
import small_gicp
from numpy.typing import NDArray


class PointClouds:
    """
    Point clouds data structure.
    """

    def __init__(
        self,
        pcd: NDArray[np.float64],
        # rgb: NDArray[np.float64],
        *,
        threads: int = 32,
    ):
        # if pcd.shape[0] != rgb.shape[0]:
        #     raise ValueError("Point cloud and RGB must have the same number of points.")

        self._pcd = small_gicp.PointCloud(pcd)
        self._kdtree = None
        self._threads = threads

        # convert to lab color space
        # self._lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        #     np.float64
        # )

    def __len__(self):
        return self._pcd.size()

    def preprocess(self, knn: int):
        """build kdtree and estimate normals and covariances."""
        self._kdtree = small_gicp.KdTree(self._pcd, num_threads=self._threads)
        small_gicp.estimate_normals_covariances(
            self._pcd, self._kdtree, num_threads=self._threads, num_neighbors=knn
        )

    # @property
    # def lab(self) -> NDArray[np.float64]:
    #     """
    #     Returns
    #     -------
    #     lab: NDArray[np.float64], shape=(n, 3)
    #         Lab color data.
    #     """
    #     return self._lab

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = small_gicp.KdTree(self._pcd, num_threads=self._threads)
        return self._kdtree

    @property
    def points(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        pcd: NDArray[np.float64], shape=(n, 4)
            Point cloud data.
        """
        return self._pcd.points()

    @property
    def covs(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        covs: NDArray[np.float64], shape=(n, 4, 4)
            Covariance matrix for each point.
        """
        return self._pcd.covs()

    @property
    def normals(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        normals: NDArray[np.float64], shape=(n, 4)
            Normal vector for each point.
        """
        return self._pcd.normals()

    def normal(self, index: int) -> NDArray[np.float64]:
        """
        Returns
        -------
        normal: NDArray[np.float64], shape=(4,) = (nx, ny, nz, 0)
            Normal vector of the point at index.
        """
        return self._pcd.normal(index)

    def cov(self, index: int) -> NDArray[np.float64]:
        """
        Returns
        -------
        cov: NDArray[np.float64], shape=(4, 4) = (3x3 matrix) + zero padding
            Covariance matrix of the point at index.
        """
        return self._pcd.cov(index)

    def point(self, index: int) -> NDArray[np.float64]:
        """
        Returns
        -------
        point: NDArray[np.float64], shape=(4,) = (x, y, z, 1)
            Point at index.
        """
        return self._pcd.point(index)
