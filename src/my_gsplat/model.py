from dataclasses import dataclass

import nerfview
import torch
from gsplat import rasterization
from kornia import geometry as KG
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from ..data.base import DEVICE
from ..data.utils import to_tensor
from .geometry import construct_full_pose, init_gs_scales
from .transform import quat_to_rotation_matrix, rotation_matrix_to_quaternion
from .utils import rgb_to_sh


# NOTE: to prevent to overwrite __hash__,which lead to failed to callmodule.named_parameters():
@dataclass(frozen=True)
class CameraConfig:
    trans_lr: float = 1e-3
    quat_lr: float = 5 * 1e-4
    quat_opt_reg: float = 1e-3
    trans_opt_reg: float = 1e-3


# NOTE: quat + tans
class CameraOptModule_quat_tans(nn.Module):
    def __init__(self, init_pose: Tensor, *, config: CameraConfig = CameraConfig()):
        super().__init__()
        self.config = config

        self.quat_cur = nn.Parameter(rotation_matrix_to_quaternion(init_pose[:3, :3]))
        self.t_cur = nn.Parameter(init_pose[:3, 3])

        # Add these lines to store previous poses
        self.prev_quat = self.quat_cur.detach().clone()
        self.prev_t = self.t_cur.detach().clone()

        self.optimizers = self._create_optimizers()

    def update_pose(self, new_pose: Tensor | None = None):
        """
        update pose with constant velocity model if new_pose is None
        or update with best pose after a train loop
        Parameters
        ----------
        new_pose: Tensor|None, shape=(4,4)
        """
        with torch.no_grad():
            if torch.is_tensor(new_pose) and new_pose.shape == (4, 4):
                self.quat_cur.data = rotation_matrix_to_quaternion(new_pose[:3, :3])
                self.t_cur.data = new_pose[:3, 3]
                self.optimizers = self._create_optimizers()
            elif new_pose is None:
                # update with constant velocity prediction
                self.quat_cur.data, self.t_cur.data = self.predict_next_pose()
            else:
                raise ValueError("fake new pose")

    def predict_next_pose(self):
        """before next loop call it to predict"""
        # Implement constant velocity model for quaternions
        quaternion_velocity = self.quat_cur - self.prev_quat
        predicted_quaternion = KG.normalize_quaternion(
            self.quat_cur + quaternion_velocity
        )

        # Implement constant velocity model for translation
        translation_velocity = self.t_cur - self.prev_t
        predicted_translation = self.t_cur + translation_velocity

        # update pre pose
        self.prev_quat, self.prev_t = (
            self.quat_cur.detach().clone(),
            self.t_cur.detach().clone(),
        )
        return predicted_quaternion, predicted_translation

    def forward(self) -> Tensor:
        cur_rotation = quat_to_rotation_matrix(self.quat_cur)
        cur_c2w = construct_full_pose(cur_rotation, self.t_cur)
        return cur_c2w

    def optimizer_step(self):
        # Perform optimization step
        for optimizer in self.optimizers:
            optimizer.step()

    def optimizer_clean(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            ("quat", self.quat_cur, self.config.quat_lr),
            ("trans", self.t_cur, self.config.trans_lr),
        ]
        optimizers = [
            Adam(
                [
                    {
                        "params": param,
                        "lr": lr,
                        "name": name,
                    }
                ],
                weight_decay=(
                    self.config.quat_opt_reg
                    if name == "quat"
                    else self.config.trans_opt_reg
                ),
            )
            for name, param, lr in params
        ]
        return optimizers


@dataclass
class GsConfig:
    init_opa: float = 1.0
    sparse_grad: bool = False
    packed: bool = False
    absgrad: bool = False
    antialiased: bool = False
    # Degree of spherical harmonics
    sh_degree: int = 1

    # RasterizeConfig
    # Near plane clipping distance
    near_plane: float = 1e-2
    # Far plane clipping distance
    far_plane: float = 1e10


class GSModel(nn.Module):
    def __init__(
        self,
        # dataset
        points: Tensor,  # N,3
        colors: Tensor,  # N,3
        *,
        # config
        config: GsConfig = GsConfig(),
    ):
        super().__init__()
        self.optimizers = None
        self.config = config
        points = points
        rgbs = colors

        # visualize_point_cloud(points, rgbs)

        # Parameters
        self.means3d = points  # [N, 3]
        self.opacities = torch.logit(
            torch.full((points.shape[0],), self.config.init_opa, device=self.device)
        )
        # [N,]
        # Calculate distances for initial scale
        self.scales = init_gs_scales(points)
        # NOTE: no deformation
        self.quats = to_tensor([1, 0, 0, 0], requires_grad=True).repeat(
            points.shape[0], 1
        )  # [N, 4]
        # self.quats = torch.nn.Parameter(quats)

        # # color is SH coefficients.
        colors = torch.zeros(
            (points.shape[0], (self.config.sh_degree + 1) ** 2, 3), device=self.device
        )  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)  # Initialize SH coefficients
        self.colors = rgbs
        self.sh0 = colors[:, :1, :]
        self.shN = colors[:, 1:, :]

    def __len__(self):
        return self.means3d.shape[0]

    def forward(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        render_mode: str = "RGB+ED",
    ):
        assert self.means3d.shape[0] == self.opacities.shape[0]
        opacities = torch.sigmoid(self.opacities)
        colors = torch.cat([self.sh0, self.shN], 1)
        # colors = torch.sigmoid(self.colors)
        scales = self.scales
        # scales = torch.exp(self.scales)

        render_colors, render_alphas, info = rasterization(
            means=self.means3d,
            quats=self.quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=self.config.sh_degree,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.config.packed,
            absgrad=self.config.absgrad,
            sparse_grad=self.config.sparse_grad,
            far_plane=self.config.far_plane,
            near_plane=self.config.near_plane,
            render_mode=render_mode,
            rasterize_mode="classic",
        )

        return render_colors, render_alphas, info

    @property
    def device(self):
        return self.means3d.device

    @torch.no_grad()
    def viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(DEVICE)
        K = torch.from_numpy(K).float().to(DEVICE)
        render_colors, _, _ = self.forward(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            # sh_degree=self.config.sh_degree,  # active all SH degrees
            # radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
