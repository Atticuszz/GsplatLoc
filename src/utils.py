import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def show_image(color, depth):
    # Plot color image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(color)
    plt.title("Color Image")
    plt.axis("off")

    # Plot depth image
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("Depth Image")
    plt.axis("off")

    plt.show()


def depth_to_colormap(depth_image: NDArray):
    # Normalize and convert the depth image to an 8-bit format, and apply a colormap
    depth_normalized = (depth_image - np.min(depth_image)) / (
        np.max(depth_image) - np.min(depth_image)
    )
    depth_8bit = np.uint8(depth_normalized * 255)
    depth_colormap = plt.cm.jet(depth_8bit)  # Using matplotlib's colormap
    return depth_colormap
