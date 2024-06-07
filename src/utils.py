import json
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# from numba import jit, prange
# from numpy.typing import NDArray


def as_intrinsics_matrix(intrinsics: list[float]) -> np.ndarray:
    """
    Get matrix representation of intrinsics.
    :param intrinsics : [fx,fy,cx,cy]
    :return: K matrix.shape=(3,3)
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def load_camera_cfg(cfg_path: str) -> dict:
    """
    Load camera configuration from YAML or Json file.
    """
    cfg_path = Path(cfg_path)
    assert cfg_path.exists(), f"File not found: {cfg_path}"
    with open(cfg_path) as file:
        if cfg_path.suffix in [".yaml", ".yml"]:
            cfg = yaml.safe_load(file)
        elif cfg_path.suffix == ".json":
            cfg = json.load(file)
        else:
            raise TypeError(f"Failed to load cfg via:{cfg_path.suffix}")
    return cfg


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


def to_tensor(
    data: NDArray[np.float64] | list[float], device, requires_grad: bool = False
) -> torch.Tensor:
    """
    Convert numpy array to pytorch tensor.
    """
    if isinstance(data, list):
        data = np.array(data)
    return torch.tensor(
        data, dtype=torch.float64, device=device, requires_grad=requires_grad
    )
