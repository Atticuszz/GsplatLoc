from pathlib import Path

import cv2
import numpy as np
import torch
from natsort import natsorted

from ..geometry import compute_depth_gt, transform_points
from ..utils import as_intrinsics_matrix, load_camera_cfg, to_tensor
from .base import AlignData, TrainData
from .Image import RGBDImage
from .normalize import normalize_2C, align_principle_axes, transform_cameras


class BaseDataset:
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


class Replica(BaseDataset):
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


# TODO: on develop
class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32
        )

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        return np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
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

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format"""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]
