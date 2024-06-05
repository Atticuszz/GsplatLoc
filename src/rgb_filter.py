import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from src.component import vis_depth_filter
from src.slam_data import Replica


# NOTE: noise maker
def add_extreme_noise(
    depth: NDArray[np.float64], outlier_prob=0.001, min_scale=0.1, max_scale=10
) -> NDArray[np.float64]:
    """
    Adds extreme scaling noise to a small percentage of the depth image's pixels to simulate outliers.

    Parameters:
    - outlier_prob (float): Probability that a pixel is an outlier.
    - min_scale (float): The minimum scaling factor for reducing depth values.
    - max_scale (float): The maximum scaling factor for increasing depth values.
    """
    # Copy the original depth array to avoid modifying the input array
    noisy_depth = np.copy(depth)

    # Determine the number of outliers
    num_outliers = int(outlier_prob * depth.size)

    # Randomly pick indices for the outliers
    outlier_indices = np.random.choice(depth.size, num_outliers, replace=False)

    # Generate random scales for each outlier
    scales = np.random.uniform(min_scale, max_scale, size=num_outliers)

    # Apply the scales to the selected indices
    np.put(noisy_depth, outlier_indices, depth.flat[outlier_indices] * scales)

    return noisy_depth


# NOTE: eval
def show_filter_diff(
    original_depth: NDArray[np.float64],
    filtered_image: NDArray[np.float64],
):
    diff_image = cv2.absdiff(original_depth, filtered_image)
    vis_depth_filter([original_depth, filtered_image, diff_image])


# NOTE: test funcs
def bilateralFilter(depth: NDArray[np.float64]):
    original_depth = depth.astype(np.float32)
    # normalize depth to 0-1 for better filtering
    normalized_depth = depth / depth.max()

    filtered_image = cv2.bilateralFilter(
        normalized_depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75
    )
    filtered_image *= depth.max()
    return filtered_image.astype(np.float64)


def medianFilter(depth: NDArray[np.float64], kernel_size=5):
    # Ensure depth is appropriate dtype for median filtering
    if depth.dtype != np.uint8:
        depth_8bit = np.clip(depth / depth.max() * 255, 0, 255).astype(np.uint8)
    else:
        depth_8bit = depth

    # Apply median blur
    filtered_image = cv2.medianBlur(depth_8bit, kernel_size)

    # Optionally convert back to original dtype
    if depth.dtype != np.uint8:
        filtered_image = (filtered_image.astype(np.float32) / 255) * depth.max()
        filtered_image = filtered_image.astype(depth.dtype)

    return filtered_image


def grad_filter(depth: NDArray[np.float64], threshold: int = 1):
    # 计算梯度
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=7)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    mask = grad_mag > threshold

    # 创建过滤后的深度图副本
    filtered_depth = np.copy(depth)

    # 将过阈值的像素设置为0或其他指定的无效值
    filtered_depth[mask] = 0  # 或者可以设为np.nan或特定的值
    show_images(depth, grad_x, grad_y, grad_mag)
    return filtered_depth


# NOTE: test run


def filter_test():
    data = Replica()
    for rbg_d in data:
        depth = rbg_d.depth
        noisy_depth = add_extreme_noise(depth)
        true_noise = cv2.absdiff(depth, noisy_depth)
        vis_depth_filter([depth, noisy_depth, true_noise])
        filter_bil = bilateralFilter(noisy_depth)

        show_filter_diff(depth, filter_bil)

        filter_median = medianFilter(noisy_depth)
        show_filter_diff(depth, filter_median)

        filter_grad = grad_filter(noisy_depth)
        show_filter_diff(depth, filter_grad)


def show_images(depth, grad_x, grad_y, grad_mag):
    # 将深度图转换为8位图像
    depth_8bit = np.clip(depth / depth.max() * 255, 0, 255).astype(np.uint8)

    # 将深度图转换为热力图
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    # 将梯度图转换为8位图像
    grad_x_8bit = np.clip(grad_x / grad_x.max() * 255, 0, 255).astype(np.uint8)
    grad_y_8bit = np.clip(grad_y / grad_y.max() * 255, 0, 255).astype(np.uint8)
    grad_mag_8bit = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)
    # 将梯度图转换为热力图
    grad_x_colormap = cv2.applyColorMap(grad_x_8bit, cv2.COLORMAP_JET)
    grad_y_colormap = cv2.applyColorMap(grad_y_8bit, cv2.COLORMAP_JET)
    grad_mag_colormap = cv2.applyColorMap(grad_mag_8bit, cv2.COLORMAP_JET)

    # 显示深度图和梯度图
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Depth Heatmap")
    plt.imshow(depth_colormap)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Gradient X Heatmap")
    plt.imshow(grad_x_colormap)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Gradient Y Heatmap")
    plt.imshow(grad_y_colormap)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Gradient Magnitude Heatmap")
    plt.imshow(grad_mag_colormap)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    filter_test()
