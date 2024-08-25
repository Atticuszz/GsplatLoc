import json
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy._typing import NDArray
from torch import Tensor

from src.data.base import DEVICE


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


def to_tensor(data, device=DEVICE, requires_grad=False, dtype=torch.float32):
    """
    Convert numpy array or list to a PyTorch tensor.
    """
    if (
        not isinstance(data, list)
        and not isinstance(data, np.ndarray)
        and not torch.is_tensor(data)
        and not isinstance(data, int)
    ):
        raise TypeError("to tensor needs list,np.ndarray or tensor")

    data = torch.as_tensor(data, dtype=dtype, device=device)  # More efficient
    return data.requires_grad_(requires_grad)


def save_tensor(data: Tensor, file_path: Path | str):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"saved tensor to {path}!")


def load_tensor(file_path: Path | str) -> Tensor:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path)
    print(f"loaded tensor from{path}!")
    return data


def as_intrinsics_matrix(intrinsics: list[float]) -> NDArray:
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
