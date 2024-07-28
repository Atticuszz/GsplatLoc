from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted

from ..geometry import compute_depth_gt, transform_points
from ..utils import as_intrinsics_matrix, load_camera_cfg, to_tensor
from .base import AlignData, TrainData
from .Image import RGBDImage
from .normalize import normalize_2C, normalize_T


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
        input_folder: Path = Path(__file__).parents[3] / "datasets/Replica",
        cfg_file: Path = Path(__file__).parents[3] / "datasets/Replica/cam_params.json",
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


class Parser(Replica):
    def __init__(self, name: str = "room0", normalize: bool = False):
        super().__init__(name=name)
        self.K = to_tensor(self.K, requires_grad=True)
        # normalize points and pose
        self.normalize = normalize

    def __getitem__(self, index: int) -> AlignData:
        assert index < len(self)
        tar, src = super().__getitem__(index), super().__getitem__(index + 1)
        # transform to world
        tar.points = transform_points(tar.pose, tar.points)
        src.points = transform_points(tar.pose, src.points)

        # NOTE: PCA
        pca_factor = torch.scalar_tensor(1.0, device=tar.points.device)
        if self.normalize:

            # NOTE: PCA
            tar, src, pca_factor = normalize_2C(tar, src)
            ks = self.K.unsqueeze(0)  # [1, 3, 3]
            h, w = src.depth.shape

            # # NOTE: normalize_points_spherical
            # tar.points, _ = normalize_points_spherical(tar.points)
            # src.points, sphere_factor = normalize_points_spherical(src.points)
            # tar.pose = adjust_pose_spherical(tar.pose, _)
            # src.pose = adjust_pose_spherical(src.pose, sphere_factor)

            # NOTE: project depth
            src.depth = (
                compute_depth_gt(
                    src.points,
                    src.colors,
                    ks,
                    c2w=tar.pose.unsqueeze(0),
                    height=h,
                    width=w,
                )
                # / pca_factor
            )  # / sphere_factor
        return AlignData(
            pca_factor=pca_factor,
            colors=tar.colors,
            pixels=(src.rgbs / 255.0).unsqueeze(0),  # [1, H, W, 3]
            tar_points=tar.points,
            src_points=src.points,
            src_depth=src.depth.unsqueeze(-1).unsqueeze(0),  # [1, H, W, 1]
            tar_c2w=tar.pose,
            src_c2w=src.pose,
            tar_nums=tar.points.shape[0],
        )


class Parser2(Replica):
    def __init__(self, name: str = "room0", normalize: bool = False):
        super().__init__(name=name)
        self.K = to_tensor(self.K, requires_grad=True)
        # normalize points and pose
        init_rgb_d: RGBDImage = super().__getitem__(0)
        init_rgb_d.points = transform_points(init_rgb_d.pose, init_rgb_d.points)
        self.normalize_T = normalize_T(init_rgb_d) if normalize else None

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: int) -> TrainData:
        assert index < len(self)
        tar = super().__getitem__(index)

        return TrainData(
            colors=tar.colors,
            pixels=tar.rgbs / 255.0,
            points=tar.points,
            depth=tar.depth,
            c2w=tar.pose,
            c2w_gt=tar.pose,
        )
