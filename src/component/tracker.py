import logging
from typing import Literal

import cv2
import numpy as np
import small_gicp
from numpy.typing import NDArray


class Scan2ScanICP:
    """
    Scan-to-Scan ICP class for depth image-derived point clouds using small_gicp library.
    Each scan is aligned with the previous scan to estimate the relative transformation.
    """

    def __init__(
        self,
        downsampling_resolution=0.05,  # Adjusted for potentially denser point clouds from depth images
        max_corresponding_distance=0.1,
        num_threads=12,
        registration_type: Literal["ICP", "PLANE_ICP", "GICP", "VGICP"] = "GICP",
        error_threshold: float = 50.0,
    ):

        self.downsampling_resolution = downsampling_resolution
        self.max_corresponding_distance = max_corresponding_distance
        self.num_threads = num_threads

        # self.max_iterations = 100
        self.previous_pcd: small_gicp.PointCloud | None = None
        self.previous_tree: small_gicp.KdTree | None = None
        self.T_last_current = np.identity(4)

        # every frame pose
        self.T_world_camera = np.identity(4)  # World to camera transformation
        self.registration_type = registration_type

        # less than min for skipping mapping, greater than as kf
        self.error_threshold = error_threshold

        # keyframe info
        # NOTE: should be in world
        self.kf_pcd: small_gicp.PointCloud | None = None
        self.kf_tree: small_gicp.KdTree | None = None
        self.kf_T_last_current = np.identity(4)

    def align_pcd(
        self,
        raw_points: NDArray[np.float64],
        init_pose: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Align new point cloud to the previous point cloud using small_gicp library.

        Parameters
        ----------
        raw_points : NDArray[np.float64]
            Current point cloud data as an (M, 3) numpy array.

        init_pose : NDArray[np.float64], optional
            Initial transformation matrix (4, 4) from world to camera, by default None.
        Returns
        -------
        T_world_camera: NDArray[np.float64]
            transformation matrix (4, 4) from world to camera.
        """
        # down sample the point cloud
        downsampled, tree = small_gicp.preprocess_points(
            raw_points, self.downsampling_resolution, num_threads=self.num_threads
        )

        # first frame
        if self.previous_pcd is None:
            self.previous_pcd = downsampled
            self.previous_tree = tree
            self.T_world_camera = init_pose if init_pose is not None else np.identity(4)
            return self.T_world_camera

        result: small_gicp.RegistrationResult = small_gicp.align(
            self.previous_pcd,
            downsampled,
            self.previous_tree,
            init_T_target_source=self.T_last_current,
            max_correspondence_distance=self.max_corresponding_distance,
            registration_type=self.registration_type,
            num_threads=self.num_threads,
            # max_iterations=self.max_iterations,
        )
        self.T_last_current = result.T_target_source
        # Update the world transformation matrix
        self.T_world_camera = self.T_world_camera @ result.T_target_source

        # Update the previous point cloud and its tree for the next iteration
        self.previous_pcd = downsampled
        self.previous_tree = tree

        return self.T_world_camera

    def keyframe(self) -> bool:
        if self.previous_pcd is None:
            raise ValueError("No previous point cloud to set as keyframe.")

        if self.kf_pcd is None:
            # update keyframe
            self.kf_pcd = self.previous_pcd
            self.kf_tree = self.previous_tree
            self.kf_T_last_current = self.T_last_current
            return True

        result: small_gicp.RegistrationResult = small_gicp.align(
            self.kf_pcd,
            self.previous_pcd,
            self.kf_tree,
            init_T_target_source=self.kf_T_last_current,
            max_correspondence_distance=self.max_corresponding_distance,
            registration_type=self.registration_type,
            num_threads=self.num_threads,
            # max_iterations=self.max_iterations,
        )

        if result.error > self.error_threshold:
            # update keyframe
            self.kf_pcd = self.previous_pcd
            self.kf_tree = self.previous_tree
            self.kf_T_last_current = self.T_last_current
            print(f"kf factor error: {result.error}")
            return True
        print(f"not kf factor error: {result.error}")
        return False


class Scan2ModelICP:
    """
    Scan-to-Model ICP class for depth image-derived point clouds using small_gicp library.
    Each scan is aligned with the previous scan to estimate the relative transformation.
    """

    def __init__(
        self,
        downsampling_resolution=0.01,  # Adjusted for potentially denser point clouds from depth images
        max_corresponding_distance=0.1,
        num_threads=14,
        registration_type: Literal["VGICP"] = "VGICP",
    ):
        self.downsampling_resolution = downsampling_resolution
        self.max_corresponding_distance = max_corresponding_distance
        self.num_threads = num_threads
        self.target: small_gicp.GaussianVoxelMap = small_gicp.GaussianVoxelMap(1.0)
        self.target.set_lru(horizon=100, clear_cycle=10)

        self.T_last_current = np.identity(4)
        self.T_world_camera = np.identity(4)  # World to camera transformation
        self.registration_type = registration_type

    def align_pcd(
        self,
        raw_points: NDArray[np.float64],
        init_pose: NDArray[np.float64] | None = None,
    ) -> np.ndarray:
        """
        Align new point cloud to the previous point cloud using small_gicp library.

        Parameters
        ----------
        raw_points : np.ndarray
            Current point cloud data as an (M, 3) numpy array.

        init_pose : np.ndarray, optional
            Initial transformation matrix (4, 4) from world to camera, by default None.
        Returns
        -------
        np.ndarray
            Updated transformation matrix (4, 4) from world to camera.
        """
        downsampled, tree = small_gicp.preprocess_points(
            raw_points, self.downsampling_resolution, num_threads=self.num_threads
        )

        if self.target.size() == 0:
            init_pose = init_pose if init_pose is not None else np.identity(4)
            self.target.insert(downsampled, init_pose)
            self.T_world_camera = init_pose
            return self.T_world_camera

        result: small_gicp.RegistrationResult = small_gicp.align(
            self.target,
            downsampled,
            init_T_target_source=self.T_world_camera @ self.T_last_current,
            max_correspondence_distance=self.max_corresponding_distance,
            # registration_type=self.registration_type,
            num_threads=self.num_threads,
        )
        self.T_last_current = (
            np.linalg.inv(self.T_world_camera) @ result.T_target_source
        )
        # Update the world transformation matrix
        self.T_world_camera = result.T_target_source

        self.target.insert(downsampled, self.T_world_camera)

        return self.T_world_camera


# NOTE: not work
class PoseEstimator:
    """To estimate the camera pose based on sequential RGB images."""

    def __init__(self, K: np.ndarray):
        self.K = K
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.first_frame = True
        self.reference_kp = None
        self.reference_des = None
        self.reference_img = None

    def add_frame(self, img: np.ndarray) -> np.ndarray:
        """Add a new frame and compute the pose transformation matrix.
        :return transform_matrix: c2w
        """
        kp, des = self.orb.detectAndCompute(img, None)
        if self.first_frame:
            # 初始化参考帧的处理
            self.reference_kp = kp
            self.reference_des = des
            self.reference_img = img
            self.first_frame = False
            return np.eye(4)

        matches = self.matcher.match(self.reference_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        logging.info(f"Found {len(matches)} matches.")

        if len(matches) < 8:
            logging.warning("Not enough matches to find a reliable pose.")
            return None

        # 匹配点处理和Essential Matrix的计算...
        if E is None:
            logging.error("Failed to compute a valid Essential Matrix.")
            return None
        src_pts = np.float32(
            [self.reference_kp[m.queryIdx].pt for m in matches]
        ).reshape(-1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # 归一化
        def normalize_points(pts):
            mean = np.mean(pts, axis=0)
            std = np.std(pts)
            return (pts - mean) / std, mean, std

        src_pts_norm, src_mean, src_std = normalize_points(src_pts)
        dst_pts_norm, dst_mean, dst_std = normalize_points(dst_pts)

        E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            return None  # 检查Essential Matrix是否成功计算

        _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.K)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t.ravel()
        scale = np.eye(3) * src_std
        scale[2, 2] = 1
        T = np.eye(4)
        T[:3, :3] = scale
        T[:3, 3] = src_mean
        transform_matrix = np.linalg.inv(T) @ transform_matrix @ T
        return transform_matrix
