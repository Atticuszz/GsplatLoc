from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from nerfview import Viewer
from torch import Tensor
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from viser import ViserServer

from ..utils import DEVICE


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
    max_steps: int = 1000
    eval_steps: list[int] = field(default_factory=lambda: [200, 30_000])
    save_steps: list[int] = field(default_factory=lambda: [1000, 30_000])
    steps_scaler: float = 1.0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000


@dataclass
class OptimizationConfig:
    ssim_lambda: float = 0.5
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

    early_stop: bool = True
    patience = 80
    best_eR = float("inf")
    best_eT = float("inf")
    best_loss = float("inf")
    best_depth_loss = float("inf")
    best_silhouette_loss = float("inf")

    counter = 0

    def init_loss(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)


@dataclass
class DepthLossConfig:
    depth_loss: bool = False
    depth_lambda: float = 0.5


@dataclass
class ViewerConfig:
    disable_viewer: bool = True
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
class Config(
    TrainingConfig,
    DatasetConfig,
    OptimizationConfig,
    DepthLossConfig,
    ViewerConfig,
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


@dataclass
class TensorWrapper:
    def to(self, device: torch.device) -> "TensorWrapper":
        """
        Move all Tensor attributes to the specified device.
        """
        for attr_name, attr_value in self.__dict__.items():
            if torch.is_tensor(attr_value):
                setattr(self, attr_name, attr_value.to(device))
        return self

    def enable_gradients(self) -> "TensorWrapper":
        """
        Enable gradients for all Tensor attributes of the TrainData instance.
        """
        for attr_name, attr_value in self.__dict__.items():
            if torch.is_tensor(attr_value):
                setattr(self, attr_name, attr_value.requires_grad_())
        return self

    def __post_init__(self):
        """
        Ensure all tensors are on the same device after initialization.
        """
        self.to(DEVICE)
        self.enable_gradients()


@dataclass
class AlignData(TensorWrapper):
    """normed data"""

    # for GS

    colors: Tensor  # N,3
    pixels: Tensor  # H,W,3
    points: Tensor  # N,3
    tar_points: Tensor
    src_points: Tensor
    src_depth: Tensor
    tar_c2w: Tensor  # 4,4
    src_c2w: Tensor  # 4,4
    tar_nums: int  # for slice tar and src
    scale_factor: Tensor  # for scale depth after rot normalized


@dataclass
class TrainData(TensorWrapper):
    """normed data"""

    # for GS
    points: Tensor  # N,3  in camera
    colors: Tensor  # N,3
    pixels: Tensor  # H,W,3

    depth: Tensor  # H,w
    c2w: Tensor  # 4,4

    scale_factor: Tensor = torch.scalar_tensor(
        1.0
    )  # for scale depth after rot normalized
