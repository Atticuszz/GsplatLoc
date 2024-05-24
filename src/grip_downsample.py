import cv2

from src.component import Scan2ScanICP
from src.component.eval import (
    WandbLogger,
    calculate_translation_error,
    calculate_rotation_error,
    calculate_pointcloud_rmse,
    diff_pcd_COM,
)
from src.slam_data import Replica, RGBDImage


def run_test(max_images: int = 2000):

    data = Replica()
    registration = Scan2ScanICP()
    logger = WandbLogger()
    for i, rgbd_image in enumerate(data):

        if i >= max_images:
            break
        # print(f"Processing image {i + 1}/{len(data)}...")
        rgbd_image: RGBDImage
        # convert tensors to numpy arrays
        if rgbd_image.pose is None:
            raise ValueError("Pose is not available.")

        start_downsample = cv2.getTickCount()
        new_pcd = rgbd_image.pointclouds(6)
        cost_time_downsample = cv2.getTickCount() - start_downsample
        logger.log_downsample_time(cost_time_downsample, i)

        start = cv2.getTickCount()
        if i == 0:
            res = registration.align_pcd(new_pcd, rgbd_image.pose)
            continue
        else:
            res = registration.align_pcd(new_pcd)
        cost_time = cv2.getTickCount() - start
        logger.log_align_time(cost_time, i)
        logger.log_align_error(res.error, i)
        logger.log_iter_times(res.iterations, i)

        est_pose = registration.T_world_camera

        eT = calculate_translation_error(est_pose, rgbd_image.pose)
        logger.log_translation_error(eT, i)

        eR = calculate_rotation_error(est_pose, rgbd_image.pose)
        logger.log_rotation_error(eR, i)

        gt_pcd = rgbd_image.camera_to_world(rgbd_image.pose, new_pcd)
        est_pcd = rgbd_image.camera_to_world(est_pose, new_pcd)
        rmse = calculate_pointcloud_rmse(est_pcd, gt_pcd)
        logger.log_rmse_pcd(rmse, i)

        com = diff_pcd_COM(est_pcd, gt_pcd)
        logger.log_com_diff(com, i)
    logger.finish()


if __name__ == "__main__":
    run_test()
