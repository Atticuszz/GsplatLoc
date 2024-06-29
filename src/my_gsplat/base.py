from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from nerfview import Viewer
from numpy.typing import NDArray
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from viser import ViserServer

from .datasets.normalize import normalize_dataset_slice, scene_scale
from .structure import Replica, RGBDImage
from .utils import DEVICE


@dataclass
class DatasetConfig:
    data_dir: str = "./data/360_v2/garden"
    data_factor: int = 4
    result_dir: str | Path = "./results/Replica"
    test_every: int = 8
    patch_size: int | None = None
    global_scale: float = 1.0

    # make_dir
    res_dir: Path | None = None
    stats_dir: Path | None = None
    render_dir: Path | None = None
    ckpt_dir: Path | None = None

    # load_data
    # parser: Parser | None = None
    trainset: list[RGBDImage] | None = None
    # valset: Dataset | None = None
    scene_scale: float | None = None

    # extra
    pcd: NDArray | None = None  # N,3
    color: NDArray | None = None  # N,3

    def load_data(self, depth_loss, normalize: bool = True):
        # Load data: Training data should contain initial points and colors.
        # self.parser = Parser(
        #     data_dir=self.data_dir,
        #     factor=self.data_factor,
        #     normalize=True,
        #     test_every=self.test_every,
        # )
        # self.trainset = Dataset(
        #     self.parser,
        #     split="train",
        #     patch_size=self.patch_size,
        #     load_depths=depth_loss,

        start = 1000
        step = 20
        self.trainset = normalize_dataset_slice(Replica()[start:start+step:8])
        print(len(self.trainset))
        # self.valset = Dataset(self.parser, split="val")
        self.scene_scale = scene_scale(self.trainset).item() * 1.1 * self.global_scale

        self.c2w_gts = []
        for rgb_d in self.trainset:
            self.c2w_gts.append(rgb_d.pose)
            # rgb_d.pose = self.trainset[0].pose

        print("Scene scale:", self.scene_scale)

    def make_dir(self):
        # Where to dump results.
        self.res_dir = Path(self.result_dir)
        self.res_dir.mkdir(exist_ok=True, parents=True)

        # Setup output directories.
        self.ckpt_dir = self.res_dir / "ckpts"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.stats_dir = self.res_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True)
        self.render_dir = self.res_dir / "renders"
        self.render_dir.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    batch_size: int = 1
    max_steps: int = 200
    eval_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    steps_scaler: float = 1.0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000


@dataclass
class OptimizationConfig:
    ssim_lambda: float = 0.2
    init_opa: float = 0.1
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    sparse_grad: bool = False
    packed: bool = False
    absgrad: bool = False
    antialiased: bool = False
    random_bkgd: bool = False
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000

    ssim: StructuralSimilarityIndexMeasure = None
    psnr: PeakSignalNoiseRatio = None
    lpips: LearnedPerceptualImagePatchSimilarity = None

    def init_loss(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)


@dataclass
class RasterizeConfig:
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10


@dataclass
class CameraConfig:
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0


@dataclass
class AppearanceConfig:
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6


@dataclass
class DepthLossConfig:
    depth_loss: bool = False
    depth_lambda: float = 1e-2


@dataclass
class ViewerConfig:
    disable_viewer: bool = False
    port: int = 8080

    # init view
    server: ViserServer | None = None
    viewer: Viewer | None = None

    def init_view(self, viewer_render_fn: Callable):
        if not self.disable_viewer:
            self.server = ViserServer(port=self.port, verbose=False)
            self.viewer = Viewer(
                server=self.server,
                render_fn=viewer_render_fn,
                mode="training",
            )


@dataclass
class TensorboardConfig:
    tb_every: int = 100
    tb_save_image: bool = False


@dataclass
class Config(
    TrainingConfig,
    DatasetConfig,
    OptimizationConfig,
    CameraConfig,
    AppearanceConfig,
    DepthLossConfig,
    ViewerConfig,
    TensorboardConfig,
    RasterizeConfig,
):
    ckpt: str | None = None

    def adjust_steps(self, factor: float = 1.0):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
