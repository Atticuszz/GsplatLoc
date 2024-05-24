import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from src.component.visualize import vis_depth_filter


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
        self.depth = depth
        self.height = depth.shape[0]
        self.width = depth.shape[1]
        self.K = K
        self.scale = depth_scale
        self.pose: NDArray[np.float64] | None = pose

    # TODO: too slow
    def camera_to_world(
        self,
        c2w: np.ndarray,
        pcd_c: NDArray[np.float64] | None = None,
        downsample_stride: int = 1,
    ) -> NDArray[np.float64]:
        """
        Transform points from camera coordinates to world coordinates using the c2w matrix.
        :param c2w: 4x4 transformation matrix from camera to world coordinates
        :param pcd_c: Nx4 numpy array of 3D points in camera coordinates
        :return: Nx4 numpy array of transformed 3D points in world coordinates
        """
        if pcd_c is None:
            points_camera = self.depth_to_pointcloud(downsample_stride)
        else:
            points_camera = pcd_c
        # Perform in-place transformation
        points_camera @= c2w.T

        return points_camera

    def depth_to_pointcloud(self, downsample_stride: int = 1) -> NDArray[np.float64]:
        """
        Convert the depth image to a 3D point cloud based on a downsample resolution.
        :param downsample_stride: Fraction of the total pixels to keep.
        :return: Nx4 numpy array of 3D points. (x,y,z,1)
        """
        # print(downsample_stride)
        self._bilateralFilter()
        # self._medianFilter()
        # Generate pixel indices
        i_indices, j_indices = np.indices(self.depth.shape)

        # Apply downsampling
        i_indices = i_indices[::downsample_stride, ::downsample_stride]
        j_indices = j_indices[::downsample_stride, ::downsample_stride]
        depth_downsampled = self.depth[::downsample_stride, ::downsample_stride]

        # Scale to meter
        depth_values = depth_downsampled.astype(np.float64) / self.scale

        # Transform to camera coordinates
        x = (j_indices - self.K[0, 2]) * depth_values / self.K[0, 0]
        y = (i_indices - self.K[1, 2]) * depth_values / self.K[1, 1]
        z = depth_values

        points = np.stack((x, y, z), axis=-1)

        # Add homogeneous coordinate
        ones = np.ones((points.shape[0], points.shape[1], 1))
        points_homogeneous = np.concatenate((points, ones), axis=-1)

        return points_homogeneous.reshape(-1, 4)

    def _bilateralFilter(self):
        original_depth = self.depth.astype(np.float32)
        normalized_depth = self.depth / self.depth.max()

        filtered_image = cv2.bilateralFilter(
            normalized_depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75
        )
        filtered_image *= self.depth.max()
        diff_image = cv2.absdiff(original_depth, filtered_image)
        vis_depth_filter([self.depth, filtered_image, diff_image])
        self.depth = filtered_image

    def _medianFilter(self, kernel_size=5):
        # Ensure self.depth is appropriate dtype for median filtering
        if self.depth.dtype != np.uint8:
            depth_8bit = np.clip(self.depth / self.depth.max() * 255, 0, 255).astype(
                np.uint8
            )
        else:
            depth_8bit = self.depth

        # Apply median blur
        filtered_image = cv2.medianBlur(depth_8bit, kernel_size)

        # Optionally convert back to original dtype
        if self.depth.dtype != np.uint8:
            filtered_image = (
                filtered_image.astype(np.float32) / 255
            ) * self.depth.max()
            filtered_image = filtered_image.astype(self.depth.dtype)

        diff_image = cv2.absdiff(self.depth, filtered_image)
        vis_depth_filter([self.depth, filtered_image, diff_image])
        # Update depth with filtered image
        self.depth = filtered_image

    # def edge_preserving_depth_adaptive(self):
    #     # 计算梯度
    #     grad_x = cv2.Sobel(self.depth, cv2.CV_64F, 1, 0, ksize=7)
    #     grad_y = cv2.Sobel(self.depth, cv2.CV_64F, 0, 1, ksize=7)
    #     grad_mag = cv2.magnitude(grad_x, grad_y)
    #     # 显示深度图和梯度图
    #
    #     self.show_images(grad_x, grad_y, grad_mag)
    #
    #     self.depth = sampled_points
    #
    #     self.show_images(grad_x, grad_y, grad_mag)
    #     n = 0
    #
    # def show_images(self, grad_x, grad_y, grad_mag):
    #     # 将深度图转换为8位图像
    #     depth_8bit = np.clip(self.depth / self.depth.max() * 255, 0, 255).astype(
    #         np.uint8
    #     )
    #
    #     # 将深度图转换为热力图
    #     depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    #
    #     # 将梯度图转换为8位图像
    #     grad_x_8bit = np.clip(grad_x / grad_x.max() * 255, 0, 255).astype(np.uint8)
    #     grad_y_8bit = np.clip(grad_y / grad_y.max() * 255, 0, 255).astype(np.uint8)
    #     grad_mag_8bit = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(
    #         np.uint8
    #     )
    #     # 将梯度图转换为热力图
    #     grad_x_colormap = cv2.applyColorMap(grad_x_8bit, cv2.COLORMAP_JET)
    #     grad_y_colormap = cv2.applyColorMap(grad_y_8bit, cv2.COLORMAP_JET)
    #     grad_mag_colormap = cv2.applyColorMap(grad_mag_8bit, cv2.COLORMAP_JET)
    #
    #     # 显示深度图和梯度图
    #     plt.figure(figsize=(15, 10))
    #
    #     plt.subplot(2, 2, 1)
    #     plt.title("Depth Heatmap")
    #     plt.imshow(depth_colormap)
    #     plt.axis("off")
    #
    #     plt.subplot(2, 2, 2)
    #     plt.title("Gradient X Heatmap")
    #     plt.imshow(grad_x_colormap)
    #     plt.axis("off")
    #
    #     plt.subplot(2, 2, 3)
    #     plt.title("Gradient Y Heatmap")
    #     plt.imshow(grad_y_colormap)
    #     plt.axis("off")
    #
    #     plt.subplot(2, 2, 4)
    #     plt.title("Gradient Magnitude Heatmap")
    #     plt.imshow(grad_mag_colormap)
    #     plt.axis("off")
    #
    #     plt.show()
