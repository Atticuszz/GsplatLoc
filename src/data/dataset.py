from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from natsort import natsorted

from src.my_gsplat.geometry import compute_depth_gt, transform_points

from .base import AlignData
from .Image import RGBDImage
from .normalize import normalize_2C
from .utils import as_intrinsics_matrix, load_camera_cfg, to_tensor


class BaseDataset(Sequence[RGBDImage]):
    def __init__(self, input_folder: str, cfg_file: str):
        assert Path(input_folder).exists(), f"Path {input_folder} does not exist."
        assert Path(cfg_file).exists(), f"Path {cfg_file} does not exist."
        self.input_folder = Path(input_folder)
        cfg_file = Path(cfg_file)
        self.cfg = load_camera_cfg(cfg_file.as_posix())["camera"]
        self.scale = self.cfg["scale"]

        self.poses = None
        self.distortion = (
            np.array(self.cfg["distortion"]) if "distortion" in self.cfg else None
        )
        self.crop_edge = self.cfg["crop_edge"] if "crop_edge" in self.cfg else 0
        if self.crop_edge:
            self.cfg["h"] -= 2 * self.crop_edge
            self.cfg["w"] -= 2 * self.crop_edge
            self.cfg["cx"] -= self.crop_edge
            self.cfg["cy"] -= self.crop_edge

        self.K = as_intrinsics_matrix(
            [self.cfg["fx"], self.cfg["fy"], self.cfg["cx"], self.cfg["cy"]]
        )

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


class Replica(BaseDataset):
    """
    Parameters:
        name: ["room" + str(i) for i in range(3)] + [
            "office" + str(i) for i in range(5)
        ]
    """

    def __init__(
        self,
        name: str = "room0",
        *,
        input_folder: Path = Path(__file__).parents[2] / "datasets/Replica",
        cfg_file: Path = Path(__file__).parents[2] / "datasets/Replica/cam_params.json",
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
        return RGBDImage(color, depth, self.K, pose)

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
        return depth / self.scale

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


class TUM(BaseDataset):
    """
    Parameters:
        name: Literal['freiburg1_desk',
                    'freiburg1_desk2','freiburg1_room',
                    'freiburg2_xyz',freiburg3_long_office_household']
    """

    def __init__(
        self,
        name: Literal[
            "freiburg1_desk",
            "freiburg1_desk2",
            "freiburg1_room",
            "freiburg2_xyz",
            "freiburg3_long_office_household",
        ] = "freiburg1_desk",
        *,
        input_folder: Path = Path(__file__).parents[2] / "datasets/TUM",
        frame_rate: int = 32,
    ):
        self.name = "rgbd_dataset_" + name
        data_dir = input_folder / self.name
        cfg_file = data_dir / "cam_params.json"
        super().__init__(data_dir.as_posix(), cfg_file.as_posix())
        self._color_paths, self._depth_paths, self._poses = self._load_tum_data(
            frame_rate
        )
        self._num_img = len(self._color_paths)

    def __str__(self):
        return f"TUM dataset: {self.name}\n in {self.input_folder}"

    def __len__(self):
        return self._num_img

    def _get_one(self, index: int) -> RGBDImage:
        color = self._get_rgb(index)
        depth = self._get_depth(index)
        pose = self._get_pose(index)
        return RGBDImage(color, depth, self.K, pose)

    def _get_pose(self, index: int | None = None) -> np.ndarray:
        return self._poses[index]

    def _get_depth(self, index: int | None = None) -> np.ndarray:
        depth_path = self._depth_paths[index]
        depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )

        if self.crop_edge > 0:
            depth = depth[
                self.crop_edge : -self.crop_edge, self.crop_edge : -self.crop_edge
            ]
        return depth / self.scale

    def _get_rgb(self, index: int | None = None) -> np.ndarray:
        color_path = self._color_paths[index]
        color = cv2.imread(color_path.as_posix(), cv2.IMREAD_COLOR)
        if self.distortion is not None:
            color = cv2.undistort(color, self.K, self.distortion)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float64)
        if self.crop_edge > 0:
            color = color[
                self.crop_edge : -self.crop_edge, self.crop_edge : -self.crop_edge
            ]
        return color

    def _load_tum_data(
        self, frame_rate: int
    ) -> tuple[list[Path], list[Path], list[np.ndarray]]:
        datapath = self.input_folder

        pose_list = datapath / (
            "groundtruth.txt"
            if (datapath / "groundtruth.txt").is_file()
            else "pose.txt"
        )
        image_list = datapath / "rgb.txt"
        depth_list = datapath / "depth.txt"

        image_data = self._parse_list(image_list)
        depth_data = self._parse_list(depth_list)
        pose_data = self._parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self._associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indices = self._get_frame_indices(associations, tstamp_image, frame_rate)

        color_paths, depth_paths, poses = [], [], []
        inv_pose = None
        for ix in indices:
            i, j, k = associations[ix]
            color_paths.append(datapath / image_data[i, 1])
            depth_paths.append(datapath / depth_data[j, 1])
            c2w = self._pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            poses.append(c2w.astype(np.float32))

        return color_paths, depth_paths, poses

    @staticmethod
    def _parse_list(filepath: Path, skiprows: int = 0) -> np.ndarray:
        return np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)

    @staticmethod
    def _associate_frames(
        tstamp_image: np.ndarray,
        tstamp_depth: np.ndarray,
        tstamp_pose: np.ndarray,
        max_dt: float = 0.08,
    ) -> list[tuple[int, int, int]]:
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))
        return associations

    @staticmethod
    def _get_frame_indices(
        associations: list[tuple[int, int, int]],
        tstamp_image: np.ndarray,
        frame_rate: int,
    ) -> list[int]:
        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices.append(i)
        return indices

    @staticmethod
    def _pose_matrix_from_quaternion(pvec: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class Parser:
    def __init__(
        self,
        data_set: Literal["Replica", "TUM"] = "Replica",
        name: str = "room0",
        normalize: bool = False,
    ):
        self._data = Replica(name) if data_set == "Replica" else TUM(name)
        self.K = to_tensor(self._data.K, requires_grad=True)
        # normalize points and pose
        self.normalize = normalize

    def __getitem__(self, index: int) -> AlignData:
        assert index < len(self._data)
        tar, src = self._data[index], self._data[index + 1]
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
                / pca_factor
            )
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
