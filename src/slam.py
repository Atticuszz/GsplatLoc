from collections import deque

import cv2
import numpy as np

from component import Mapper
from component.tracker import Scan2ScanICP
from slam_data import Camera, Replica, RGBDImage


class Slam2D:

    def __init__(self, input_folder, cfg_file: str, use_camera: bool = False) -> None:
        if use_camera:
            self.slam_data = Camera(input_folder, cfg_file)
        else:
            self.slam_data = Replica(input_folder, cfg_file)
        self.use_camera = use_camera
        self.tracker = Scan2ScanICP()
        self.mapper = Mapper()
        # fps
        self.stamps = deque(maxlen=100)
        # self.vis = PcdVisualizer(intrinsic_matrix=self.slam_data.K)

        # for vis
        self.gt_poses = []
        self.estimated_poses = []

    def run(self) -> None:
        """
        tracking and mapping
        """

        for i, rgb_d in enumerate(self.slam_data):
            rgb_d: RGBDImage
            self.gt_poses.append(rgb_d.pose)

            start = cv2.getTickCount()
            pcd_c = rgb_d.depth_to_pointcloud(8)
            if i == 0:
                pose = self.tracker.align_pcd(pcd_c, rgb_d.pose)
            else:
                pose = self.tracker.align_pcd(pcd_c)

            # pose = rgb_d.pose

            kf = self.tracker.keyframe()
            if kf:
                pcd_w = rgb_d.camera_to_world(pose, pcd_c)
                self.mapper.build_map_2d(pcd_w)
                self.mapper.show()
                # self.vis.update_render(pcd_c, pose)
                # self.vis.vis_trajectory(
                #     gt_poses=self.gt_poses,
                #     estimated_poses=self.estimated_poses,
                #     downsampling_resolution=5,
                #     fps=fps,
                # )
            end = cv2.getTickCount()
            self.stamps.append((end - start) / cv2.getTickFrequency())

            self.estimated_poses.append(pose)
            fps = 1 / np.mean(self.stamps)
            print(f"Average FPS: {fps}")

        self.mapper.show()
        # self.vis.close()
