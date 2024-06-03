from typing import Literal

import numpy as np
import small_gicp
from numpy.typing import NDArray

from src.gicp.optimizer import lm_optimize
from src.gicp.pcd import PointClouds


# TODO: add registration base class and registration result class


class GICP:
    def __init__(self):
        self.previous_pcd: PointClouds | None = None
        # every frame pose
        self.T_world_camera = np.identity(4)  # World to camera transformation

    def align_pcd_gt_pose(
        self,
        raw_points: NDArray[np.float64],
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
    ):
        raw_points = PointClouds(raw_points, np.zeros_like(raw_points))
        raw_points.preprocess(20)
        # first frame
        if self.previous_pcd is None:
            self.previous_pcd = raw_points
            self.T_world_camera = (
                init_gt_pose if init_gt_pose is not None else np.identity(4)
            )
            return init_gt_pose

        result = lm_optimize(T_last_current, self.previous_pcd, raw_points)
        # Update the world transformation matrix
        self.T_world_camera = self.T_world_camera @ result
        return result


class Scan2ScanICP:
    """
    Scan-to-Scan ICP class for depth image-derived point clouds using small_gicp library.
    Each scan is aligned with the previous scan to estimate the relative transformation.
    """

    def __init__(
        self,
        voxel_downsampling_resolutions=0.01,  # Adjusted for potentially denser point clouds from depth images
        max_corresponding_distance=0.1,
        num_threads=32,
        registration_type: Literal["ICP", "PLANE_ICP", "GICP", "VGICP"] = "GICP",
        error_threshold: float = 50.0,
    ):

        self.voxel_downsampling_resolutions = voxel_downsampling_resolutions
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
        # ) -> NDArray[np.float64]:
    ) -> small_gicp.RegistrationResult:
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
            raw_points,
            self.voxel_downsampling_resolutions,
            num_threads=self.num_threads,
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

        # return self.T_world_camera
        return result

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

    #
    def align_pcd_gt_pose(
        self,
        raw_points: NDArray[np.float64],
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
        knn: int = 10,
    ):
        # down sample the point cloud
        downsampled, tree = small_gicp.preprocess_points(
            raw_points,
            self.voxel_downsampling_resolutions,
            num_threads=self.num_threads,
            num_neighbors=knn,
        )

        # first frame
        if self.previous_pcd is None:
            self.previous_pcd = downsampled
            self.previous_tree = tree
            self.T_world_camera = (
                init_gt_pose if init_gt_pose is not None else np.identity(4)
            )
            return init_gt_pose

        result: small_gicp.RegistrationResult = small_gicp.align(
            self.previous_pcd,
            downsampled,
            self.previous_tree,
            init_T_target_source=T_last_current,
            max_correspondence_distance=self.max_corresponding_distance,
            registration_type=self.registration_type,
            num_threads=self.num_threads,
            # max_iterations=self.max_iterations,
        )

        # Update the world transformation matrix
        self.T_world_camera = self.T_world_camera @ result.T_target_source

        # Update the previous point cloud and its tree for the next iteration
        self.previous_pcd = downsampled
        self.previous_tree = tree

        return self.T_world_camera
        # return result
