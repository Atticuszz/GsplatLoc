import pprint
from typing import Literal, NamedTuple

import numpy as np

from src.component import Scan2ScanICP
from src.eval.logger import WandbLogger
from src.eval.utils import (
    calculate_pointcloud_rmse,
    calculate_rotation_error,
    calculate_translation_error,
    diff_pcd_COM,
)
from src.gicp.depth_loss import train_model, DEVICE
from src.slam_data import Replica, RGBDImage
from src.slam_data.dataset import DataLoaderBase
from src.utils import to_tensor


class RegistrationConfig(NamedTuple):
    max_corresponding_distance: float = 0.1
    num_threads: int = 32
    registration_type: Literal["ICP", "PLANE_ICP", "GICP", "COLORED_ICP"] = ("GICP",)
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
    implementation: str | None = None
    num_iters: int | None = None
    learning_rate: float | None = None

    def as_dict(self):
        return {key: val for key, val in self._asdict().items() if val is not None}


class ExperimentBase:

    def __init__(self, backends, wandb_config: WandbConfig, extra_config: dict = None):
        self.data: DataLoaderBase = Replica(wandb_config.sub_set)
        self.backends = backends
        if extra_config is None:
            extra_config = {}
        wandb_config = wandb_config.as_dict()
        wandb_config.update(**extra_config)
        pprint.pprint(wandb_config)
        self.logger = WandbLogger(config=wandb_config)

    def run(self, max_images: int = 2000):
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

        # self.grid_downsample = registration_config.grid_downsample_resolution
        # self.knn = registration_config.knn

    def run(self, max_images: int = 2000):

        pre_pose = None
        for i, rgbd_image in enumerate(self.data):

            if i >= max_images:
                break
            # print(f"Processing image {i + 1}/{len(data)}...")
            rgbd_image: RGBDImage
            # convert tensors to numpy arrays
            if rgbd_image.pose is None:
                raise ValueError("Pose is not available.")
            pre_pose = rgbd_image.pose

            if self.backends.registration_type == "COLORED_ICP":
                new_pcd = rgbd_image.color_pcds(True)
            else:
                new_pcd = rgbd_image.color_pcds()

            # NOTE: align interface
            if i == 0:
                # res = self.backends.align(new_pcd, rgbd_image.pose)
                res = self.backends.align(new_pcd, rgbd_image.pose)
                continue
            else:
                T_last_current = rgbd_image.pose @ np.linalg.inv(pre_pose)
                # res = self.backends.align(new_pcd, T_last_current, knn=self.knn)
                res = self.backends.align(new_pcd, T_last_current)

            # NOTE: align data
            # self.logger.log_align_error(res.error, i)
            # self.logger.log_iter_times(res.iterations, i)
            # NOTE: eT
            est_pose = self.backends.T_world_camera
            eT = calculate_translation_error(est_pose, rgbd_image.pose)
            self.logger.log_translation_error(eT, i)
            # NOTE:ER
            eR = calculate_rotation_error(est_pose, rgbd_image.pose)
            self.logger.log_rotation_error(eR, i)
            # NOTE:RMSE
            gt_pcd = rgbd_image.camera_to_world(rgbd_image.pose, new_pcd)
            est_pcd = rgbd_image.camera_to_world(est_pose, new_pcd)
            rmse = calculate_pointcloud_rmse(est_pcd, gt_pcd)
            self.logger.log_rmse_pcd(rmse, i)
            # NOTE:COM
            com = diff_pcd_COM(est_pcd, gt_pcd)
            self.logger.log_com_diff(com, i)
        self.logger.finish()


class DepthLossExperiment(ExperimentBase):

    def __init__(self, wandb_config: WandbConfig):
        super().__init__(
            backends=train_model, wandb_config=wandb_config, extra_config=kwargs
        )
        self.num_iters = wandb_config.num_iters
        self.learning_rate = wandb_config.learning_rate

    def run(self, max_images: int = 2000):

        for i, rgbd_image in enumerate(self.data):

            if i >= max_images:
                break
            rgbd_image: RGBDImage
            if rgbd_image.pose is None:
                raise ValueError("Pose is not available.")

            new_pcd = rgbd_image.color_pcds()

            # NOTE: align interface
            if i == 0:
                pose = rgbd_image.pose
                continue
            else:

                min_loss, pose = self.backends(
                    self.data[i - 1],
                    rgbd_image,
                    to_tensor(self.data.K, device=DEVICE),
                    num_iterations=self.num_iters,
                    learning_rate=self.learning_rate,
                )

            # NOTE: loss
            self.logger.log_loss(min_loss, i)

            # NOTE: eT
            est_pose = pose.detach().cpu().numpy()
            eT = calculate_translation_error(est_pose, rgbd_image.pose)
            self.logger.log_translation_error(eT, i)
            # NOTE:ER
            eR = calculate_rotation_error(est_pose, rgbd_image.pose)
            self.logger.log_rotation_error(eR, i)
            # NOTE:RMSE
            gt_pcd = rgbd_image.camera_to_world(rgbd_image.pose, new_pcd)
            est_pcd = rgbd_image.camera_to_world(est_pose, new_pcd)
            rmse = calculate_pointcloud_rmse(est_pcd, gt_pcd)
            self.logger.log_rmse_pcd(rmse, i)
            # NOTE:COM
            com = diff_pcd_COM(est_pcd, gt_pcd)
            self.logger.log_com_diff(com, i)
        self.logger.finish()
