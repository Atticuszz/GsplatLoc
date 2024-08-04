import pprint
from typing import Literal, NamedTuple

import numpy as np

from src.component import Scan2ScanICP
from src.data.dataset import Replica
from src.data.Image import RGBDImage
from src.eval.logger import WandbLogger
from src.eval.utils import calculate_rotation_error_np, calculate_translation_error_np

# from src.pose_estimation.train_eval import train_model_with_adam, train_model_with_LBFGS


class RegistrationConfig(NamedTuple):
    max_corresponding_distance: float = 0.1
    num_threads: int = 32
    registration_type: Literal["ICP", "PLANE_ICP", "GICP", "COLORED_ICP"] = "GICP"
    voxel_downsampling_resolutions: float | None = None
    # grid_downsample_resolution: int | None = None
    # for gicp estimate normals and covs 10 is the best after tests
    # knn: int = 10

    def as_dict(self):
        return {key: val for key, val in self._asdict().items() if val is not None}


class WandbConfig(NamedTuple):
    algorithm: str = "GICP"
    dataset: str = "Replica"
    sub_set: str = "office0"
    description: str = "GICP on Replica dataset"
    normalize: bool = True
    implementation: str | None = None
    num_iters: int | None = None
    learning_rate: float | None = None
    optimizer: str | None = None

    def as_dict(self):
        return {key: val for key, val in self._asdict().items() if val is not None}


class ExperimentBase:

    def __init__(
        self, wandb_config: WandbConfig, extra_config: dict = None, backends=None
    ):
        # self.data: DataLoaderBase = Replica(wandb_config.sub_set)
        self.sub_set = wandb_config.sub_set
        self.backends = backends
        if extra_config is None:
            extra_config = {}
        wandb_config = wandb_config.as_dict()
        wandb_config.update(**extra_config)
        pprint.pprint(wandb_config)

        self.logger = WandbLogger(config=wandb_config)

    def run(self):
        raise NotImplementedError


class ICPExperiment(ExperimentBase):
    def __init__(
        self,
        registration_config: RegistrationConfig,
        wandb_config: WandbConfig,
    ):
        super().__init__(
            backends=Scan2ScanICP(**registration_config.as_dict()),
            wandb_config=wandb_config,
            extra_config=registration_config.as_dict(),
        ),
        self.data = Replica(wandb_config.sub_set)
        # self.grid_downsample = registration_config.grid_downsample_resolution
        # self.knn = registration_config.knn

    def run(self, max_images: int = 2000):

        pre_pose = None
        for i, rgbd_image in enumerate(self.data):

            # print(f"Processing image {i + 1}/{len(data)}...")
            rgbd_image: RGBDImage
            # convert tensors to numpy arrays
            if rgbd_image.pose is None:
                raise ValueError("Pose is not available.")
            pre_pose = rgbd_image.pose.cpu().numpy()
            pose_gt = rgbd_image.pose.cpu().numpy()
            if self.backends.registration_type == "COLORED_ICP":
                new_pcd = np.concatenate(
                    (rgbd_image.points.cpu().numpy(), rgbd_image.colors.cpu().numpy()),
                    axis=1,
                )
            else:
                new_pcd = rgbd_image.points.cpu().numpy()

            # NOTE: align interface
            if i == 0:
                # res = self.backends.align(new_pcd, rgbd_image.pose)
                res = self.backends.align(new_pcd, pose_gt)
                continue
            else:
                T_last_current = pose_gt @ np.linalg.inv(pre_pose)
                self.backends.T_world_camera = pre_pose
                # res = self.backends.align(new_pcd, T_last_current, knn=self.knn)
                res = self.backends.align(new_pcd, T_last_current)

            # NOTE: align data
            # self.logger.log_align_error(res.error, i)
            # self.logger.log_iter_times(res.iterations, i)
            # NOTE: eT
            est_pose = self.backends.T_world_camera
            eT = calculate_translation_error_np(est_pose, pose_gt)
            self.logger.log_translation_error(eT, i)
            # NOTE:ER
            eR = calculate_rotation_error_np(est_pose, pose_gt)
            self.logger.log_rotation_error(eR, i)
            # # NOTE:RMSE
            # gt_pcd = rgbd_image.camera_to_world(rgbd_image.pose, new_pcd)
            # est_pcd = rgbd_image.camera_to_world(est_pose, new_pcd)
            # rmse = calculate_pointcloud_rmse(est_pcd, gt_pcd)
            # self.logger.log_rmse_pcd(rmse, i)
            # # NOTE:COM
            # com = diff_pcd_COM(est_pcd, gt_pcd)
            # self.logger.log_com_diff(com, i)
            if i >= max_images - 1:
                break
        self.logger.finish()
