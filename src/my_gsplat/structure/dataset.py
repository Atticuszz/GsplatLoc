from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

from ..utils import as_intrinsics_matrix, load_camera_cfg
from .Image import RGBDImage


class DataLoaderBase:
    def __init__(self, input_folder: str, cfg_file: str):
        assert Path(input_folder).exists(), f"Path {input_folder} does not exist."
        assert Path(cfg_file).exists(), f"Path {cfg_file} does not exist."
        self.input_folder = Path(input_folder)
        cfg_file = Path(cfg_file)
        self.cfg = load_camera_cfg(cfg_file.as_posix())["camera"]
        self.scale = self.cfg["scale"]
        self.K = as_intrinsics_matrix(
            [self.cfg["fx"], self.cfg["fy"], self.cfg["cx"], self.cfg["cy"]]
        )
        self.poses = None
        self.cur = 0

    def __len__(self):
        """get dataset num"""
        raise NotImplementedError

    def __getitem__(self, index: int) -> list[RGBDImage] | RGBDImage:
        if isinstance(index, int):
            if index >= len(self) or index < 0:
                raise ValueError(f"Index {index} out of range (0 to {len(self) - 1})")
            return self._get_one(index)
        elif isinstance(index, slice):

            return [self._get_one(i) for i in range(*index.indices(len(self)))]
        else:
            raise TypeError(f"index must be int or slice but now is {type(index)}")

    def _get_one(self, index: int) -> RGBDImage:
        raise NotImplementedError

    def _get_rgb(self, index: int | None = None) -> np.ndarray:
        """
        :return: rgb_frame.shape=(height,width,color)
        """
        raise NotImplementedError

    def _get_depth(self, index: int | None = None) -> np.ndarray:
        """
        :return: depth_array.shape = (height,width)
        """
        raise NotImplementedError

    def _get_pose(self, index: int | None = None) -> np.ndarray:
        """
        :return: c2w: 4x4 transformation matrix from camera to world coordinates
        """
        raise NotImplementedError


class Replica(DataLoaderBase):
    def __init__(
        self,
        name: str = "room0",
        *,
        input_folder: Path = Path(__file__).parents[3] / "Datasets/Replica",
        cfg_file: Path = Path(__file__).parents[3] / "Datasets/Replica/cam_params.json",
    ):
        self.name = name
        super().__init__((input_folder / name).as_posix(), cfg_file.as_posix())
        self._color_paths, self._depth_paths = self._filepaths()
        self._num_img = len(self._color_paths)
        self._poses = self._load_poses()

    def __str__(self):
        return f"Replica dataset: {self.name}\n in {self.input_folder}"

    def __len__(self):
        return self._num_img

    def _get_one(self, index: int) -> RGBDImage:
        color = self._get_rgb(index)
        depth = self._get_depth(index)
        pose = self._get_pose(index)
        return RGBDImage(color, depth, self.K, self.scale, pose)

    def _get_pose(self, index: int | None = None) -> np.ndarray:
        pose = self._poses[index]
        # pose[:3, 3] *= self.scale
        return pose

    def _get_depth(self, index: int | None = None) -> np.ndarray:
        depth_path = self._depth_paths[index]
        if depth_path.suffix == ".png":
            depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(
                np.float64
            )
        else:
            raise ValueError(f"Unsupported depth file format {depth_path.suffix}.")
        return depth

    def _get_rgb(self, index: int | None = None) -> np.ndarray:
        color_path = self._color_paths[index]
        # convert to rgb
        color = cv2.imread(color_path.as_posix(), cv2.IMREAD_COLOR).astype(np.float64)
        return color

    def _load_poses(self, index: int | None = None) -> list[np.ndarray]:
        """
        c2w: 4x4 transformation matrix from camera to world coordinates
        :return: list[pose]
        """
        poses = []
        pose_path = self.input_folder / "traj.txt"
        with open(pose_path) as f:
            lines = f.readlines()
        for i in range(self._num_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            poses.append(c2w)
        return poses

    def _filepaths(self) -> tuple[list[Path], list[Path]]:
        """get color and depth image paths"""
        color_paths = natsorted(self.input_folder.rglob("frame*.jpg"))
        depth_paths = natsorted(self.input_folder.rglob("depth*.png"))
        if len(color_paths) == 0 or len(depth_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.input_folder}. Please check the path."
            )
        elif len(color_paths) != len(depth_paths):
            raise ValueError(
                f"Number of color and depth images do not match in {self.input_folder}."
            )
        return color_paths, depth_paths
