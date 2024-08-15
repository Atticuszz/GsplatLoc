from collections.abc import Callable
from dataclasses import dataclass

import torch
from nerfview import Viewer
from torch import Tensor
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from viser import ViserServer

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {DEVICE} DEVICE")


@dataclass
class OptimizationConfig:
    max_steps: int = 1000

    ssim_lambda: float = 0.5
    depth_lambda: float = 0.8
    normal_lambda: float = 0.0

    ssim: StructuralSimilarityIndexMeasure = None
    psnr: PeakSignalNoiseRatio = None
    lpips: LearnedPerceptualImagePatchSimilarity = None

    early_stop: bool = True
    patience = 200
    best_eR = float("inf")
    best_eT = float("inf")
    best_loss = float("inf")
    best_depth_loss = float("inf")
    best_silhouette_loss = float("inf")
    best_pose: Tensor = torch.eye(4)

    counter = 0

    def init_loss(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)


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
class Config(
    OptimizationConfig,
    ViewerConfig,
):
    ckpt: str | None = None

    def adjust_steps(self, factor: float = 1.0):
        self.max_steps = int(self.max_steps * factor)


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
    pixels: Tensor  # [1, H, W, 3]
    # points: Tensor  # N,3
    tar_points: Tensor
    src_points: Tensor
    src_depth: Tensor  # B,H,w,1
    tar_c2w: Tensor  # 4,4
    src_c2w: Tensor  # 4,4
    tar_nums: int  # for slice tar and src
    # sphere_factor: Tensor  # for scale depth after rot normalized
    pca_factor: Tensor


@dataclass
class TrainData(TensorWrapper):
    """normed data"""

    # for GS
    colors: Tensor  # N,3
    pixels: Tensor  # [1, H, W, 3]

    depth: Tensor  # [1, H, W, 1]
    c2w: Tensor  # 4,4
    image_id: Tensor
    # pca_factor: Tensor = torch.scalar_tensor(
    #     1.0
    # )  # for scale depth after rot normalized
