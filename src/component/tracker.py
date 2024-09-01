from typing import Literal

import numpy as np
import open3d as o3d
import small_gicp
from numpy.typing import NDArray


class Scan2ScanICP:
    """
    Scan-to-Scan ICP class for depth image-derived point clouds using small_gicp library.
    Each scan is aligned with the previous scan to estimate the relative transformation.
    """

    def __init__(
        self,
        max_corresponding_distance=0.1,
        voxel_downsampling_resolutions: float = 0.0,
        knn: int = 20,
        num_threads=32,
        registration_type: Literal[
            "ICP", "PLANE_ICP", "GICP", "COLORED_ICP", "HYBRID"
        ] = "GICP",
        implementation: Literal["small_gicp", "open3d"] = "small_gicp",
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
        self.backend = implementation

        # less than min for skipping mapping, greater than as kf
        self.error_threshold = error_threshold

        # keyframe info
        # NOTE: should be in world
        self.kf_pcd: small_gicp.PointCloud | None = None
        self.kf_tree: small_gicp.KdTree | None = None
        self.kf_T_last_current = np.identity(4)

        self.knn = knn

        if self.backend == "open3d" and self.registration_type == "HYBRID":
            self.o3d_device = o3d.core.Device("CUDA:0")
            self.criteria_list = [
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            ]
            self.last_rgbd = None
            self.max_depth = 100.0
            self.depth_scale = 1.0

    # NOTE: following is the align_pcd_gt_pose with true T_last_current estimate
    def align(
        self,
        raw_points: NDArray[np.float64],
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
    ):
        """
        Parameters
        ----------
        raw_points: shape = (h*w*(3/4/6/7)
        """
        if self.backend == "small_gicp" and self.registration_type != "COLORED_ICP":
            return self.align_small_gicp(
                raw_points, init_gt_pose, T_last_current, self.knn
            )
        elif self.backend == "open3d":
            return self.align_o3d(raw_points, init_gt_pose, T_last_current, self.knn)
        else:
            raise ValueError("wrong backend type")

    def align_small_gicp(
        self,
        raw_points: NDArray[np.float64],
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
        knn: int = 20,
    ):
        # down sample the point cloud
        if self.voxel_downsampling_resolutions > 0.0:
            downsampled, tree = small_gicp.preprocess_points(
                raw_points,
                self.voxel_downsampling_resolutions,
                num_threads=self.num_threads,
                num_neighbors=knn,
            )
        elif self.voxel_downsampling_resolutions == 0.0:
            raw_points = small_gicp.PointCloud(raw_points)
            tree = small_gicp.KdTree(raw_points, num_threads=self.num_threads)
            small_gicp.estimate_normals_covariances(
                raw_points, tree, num_neighbors=knn, num_threads=self.num_threads
            )
            downsampled = raw_points
        else:
            raise ValueError("voxel_downsampling_resolutions must greater than 0.0")
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

    def align_o3d(
        self,
        raw_points: NDArray[np.float64],
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
        knn: int = 20,
    ):

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points[:, :3])  # 假设前三列是 XYZ
        if self.registration_type == "COLORED_ICP":
            pcd.colors = o3d.utility.Vector3dVector(raw_points[:, 4:])
            # Compute normals for the point cloud, which are needed for GICP and Colored ICP
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))

        # 体素下采样
        if self.voxel_downsampling_resolutions > 0.0:
            downsampled = pcd.voxel_down_sample(self.voxel_downsampling_resolutions)
        elif self.voxel_downsampling_resolutions == 0.0:
            downsampled = pcd
        else:
            raise ValueError("voxel_downsampling_resolutions must greater than 0.0")

        # 如果是第一帧，初始化
        if self.previous_pcd is None:
            self.previous_pcd = downsampled
            self.T_world_camera = (
                init_gt_pose if init_gt_pose is not None else np.identity(4)
            )
            return self.T_world_camera
        # selection icps
        if self.registration_type == "GICP":
            registration_method = (
                o3d.pipelines.registration.registration_generalized_icp
            )
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
            )
        elif self.registration_type == "ICP":
            registration_method = o3d.pipelines.registration.registration_icp
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        elif self.registration_type == "PLANE_ICP":
            registration_method = o3d.pipelines.registration.registration_icp
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        elif self.registration_type == "COLORED_ICP":
            registration_method = o3d.pipelines.registration.registration_colored_icp
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationForColoredICP()
            )
        else:
            raise ValueError("Unsupported registration type")

        reg = registration_method(
            source=downsampled,
            target=self.previous_pcd,
            max_correspondence_distance=self.max_corresponding_distance,
            init=T_last_current,
            estimation_method=estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
            ),
        )
        self.T_world_camera = self.T_world_camera @ reg.transformation
        self.previous_pcd = downsampled

        return self.T_world_camera

    def align_o3d_hybrid(
        self,
        image: NDArray[np.float64],
        depth: NDArray[np.float64],
        Ks: NDArray,
        init_gt_pose: NDArray[np.float64] | None = None,
        T_last_current: NDArray[np.float64] = np.identity(4),
    ):

        rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(image)).to(self.o3d_device),
            o3d.t.geometry.Image(np.ascontiguousarray(depth)).to(self.o3d_device),
        )

        if self.last_rgbd is None:
            self.last_rgbd = rgbd
            self.T_world_camera = (
                init_gt_pose if init_gt_pose is not None else np.identity(4)
            )
            return self.T_world_camera

        rel_transform = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            self.last_rgbd,
            rgbd,
            o3d.core.Tensor(Ks, dtype=o3d.core.Dtype.Float64),
            o3d.core.Tensor(T_last_current),
            self.depth_scale,
            self.max_depth,
            self.criteria_list,
            o3d.t.pipelines.odometry.Method.Hybrid,
        )

        # Adjust for the coordinate system difference
        rel_transform = rel_transform.transformation.cpu().numpy()
        rel_transform[0, [1, 2, 3]] *= -1
        rel_transform[1, [0, 2, 3]] *= -1
        rel_transform[2, [0, 1, 3]] *= -1

        self.T_world_camera = self.T_world_camera @ rel_transform
        self.last_rgbd = rgbd.clone()

        return self.T_world_camera
