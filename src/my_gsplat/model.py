import math
from dataclasses import dataclass

import kornia.geometry
import nerfview
import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor, nn
from torch.optim import Adam, Optimizer, SparseAdam
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from .datasets.base import AlignData

from .geometry import (
    rotation_6d_to_matrix,
    quaternion_to_rotation_matrix,
    construct_full_pose,
    matrix_to_rotation_6d,
)
from .utils import DEVICE, knn, normalized_quat_to_rotmat, rgb_to_sh, to_tensor
from .geometry import rotation_matrix_to_quaternion


class _CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


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
        self.c2w_start = init_pose
        self.quaternion_cur = nn.Parameter(
            rotation_matrix_to_quaternion(init_pose[:3, :3])
        )
        self.translation_cur = nn.Parameter(init_pose[:3, 3])
        self.optimizers = self._create_optimizers()

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """

        Parameters
        ----------
        points: src point ,N,3

        Returns
        -------
            tuple[new_points,cur_c2w]
        """
        cur_rotation = quaternion_to_rotation_matrix(self.quaternion_cur)
        cur_c2w = construct_full_pose(cur_rotation, self.translation_cur)
        transform_matrix = cur_c2w @ torch.linalg.inv(self.c2w_start)
        new_points = kornia.geometry.transform_points(
            transform_matrix.unsqueeze(0), points
        )
        return new_points, cur_c2w

    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            # ("means3d", self.means3d, self.lr_means3d),
            ("quat", self.quaternion_cur, self.config.quat_lr),
            ("trans", self.translation_cur, self.config.trans_lr),
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


# NOTE: 6d + tans
class CameraOptModule_6d_tans(nn.Module):
    def __init__(self, init_pose: Tensor, *, config: CameraConfig = CameraConfig()):
        super().__init__()
        self.config = config
        self.c2w_start = init_pose
        self.rotation_6d = nn.Parameter(matrix_to_rotation_6d(init_pose[:3, :3]))
        self.translation_cur = nn.Parameter(init_pose[:3, 3])
        self.optimizers = self._create_optimizers()

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """

        Parameters
        ----------
        points: src point ,N,3

        Returns
        -------
            tuple[new_points,cur_c2w]
        """
        cur_rotation = rotation_6d_to_matrix(self.rotation_6d)
        cur_c2w = construct_full_pose(cur_rotation, self.translation_cur)
        transform_matrix = cur_c2w @ torch.linalg.inv(self.c2w_start)
        new_points = kornia.geometry.transform_points(
            transform_matrix.unsqueeze(0), points
        )
        return new_points, cur_c2w

    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            ("rot_6d", self.rotation_6d, self.config.quat_lr),
            ("trans", self.translation_cur, self.config.trans_lr),
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


# NOTE: 6d_inc, tans_inc
class CameraOptModule_6d_inc_tans_inc(nn.Module):
    def __init__(
        self, init_pose: torch.Tensor, *, config: CameraConfig = CameraConfig()
    ):
        super().__init__()
        self.config = config
        self.c2w_start = init_pose

        # Identity rotation in 6D representation
        self.register_buffer(
            "identity_rotation_6d", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )

        # Initialize increments as zeros
        self.rotation_6d_inc = nn.Parameter(torch.zeros(6))
        self.translation_inc = nn.Parameter(torch.zeros(3))
        self.optimizers = self._create_optimizers()

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Adjust rotation increment by the identity rotation
        adjusted_rotation_6d = self.rotation_6d_inc + self.identity_rotation_6d
        rot_inc_matrix = rotation_6d_to_matrix(adjusted_rotation_6d)

        # Apply increments to initial pose
        init_rotation = self.c2w_start[:3, :3]
        init_translation = self.c2w_start[:3, 3]

        cur_rotation = rot_inc_matrix @ init_rotation
        cur_translation = init_translation + self.translation_inc

        cur_c2w = construct_full_pose(cur_rotation, cur_translation)
        transform_matrix = cur_c2w @ torch.linalg.inv(self.c2w_start)

        new_points = kornia.geometry.transform_points(
            transform_matrix.unsqueeze(0), points
        )
        return new_points, cur_c2w

    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            ("rot_6d", self.rotation_6d_inc, self.config.quat_lr),
            ("trans", self.translation_inc, self.config.trans_lr),
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


@dataclass(frozen=True)
class GsConfig:
    init_opa: float = 1.0
    sparse_grad: bool = False
    packed: bool = False
    absgrad: bool = False
    antialiased: bool = False
    # Degree of spherical harmonics
    sh_degree: int = 3


class GSModel(nn.Module):
    def __init__(
        self,
        # dataset
        gs_data: AlignData,
        *,
        # config
        config: GsConfig = GsConfig(),
        batch_size: int = 1,
        feature_dim: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        points = gs_data.points
        rgbs = gs_data.colors
        scene_scale = gs_data.scene_scale

        # Calculate distances for initial scale
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)

        # Parameters
        self.means3d = points  # [N, 3]
        # self.means3d = nn.Parameter(points)  # [N, 3]
        self.scales = nn.Parameter(scales)
        self.opacities = nn.Parameter(
            torch.logit(
                torch.full((points.shape[0],), self.config.init_opa, device=DEVICE)
            )
        )  # [N,]
        # NOTE: no deformation
        self.quats = torch.nn.Parameter(
            to_tensor([1, 0, 0, 0], requires_grad=True).repeat(points.shape[0], 1)
        )  # [N, 4]
        # self.quats = torch.nn.Parameter(quats)

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros(
                (points.shape[0], (self.config.sh_degree + 1) ** 2, 3), device=DEVICE
            )  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)  # Initialize SH coefficients
            self.colors = nn.Parameter(colors)
            self.sh0 = torch.nn.Parameter(colors[:, :1, :])
            self.shN = torch.nn.Parameter(colors[:, 1:, :])
        else:
            # features will be used for appearance and view-dependent shading
            self.features = nn.Parameter(torch.rand(points.shape[0], feature_dim))
            self.colors = nn.Parameter(torch.logit(rgbs))

        # Learning rates (not parameters, stored for optimizer setup)
        self.lr_means3d = 1.6e-4 * scene_scale
        self.lr_scales = 5e-3
        self.lr_opacities = 5e-2
        self.lr_features = 2.5e-3 if feature_dim is not None else None
        self.lr_colors = 2.5e-3
        self.lr_sh0 = 2.5e-3
        self.lr_shN = 2.5e-3 / 20
        self.optimizers = self._create_optimizers()

        # Running stats for prunning & growing.
        n_gauss = points.shape[0]
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=DEVICE),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=DEVICE, dtype=torch.int),
        }

    def __len__(self):
        return self.means3d.shape[0]

    def forward(
        self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int, **kwargs
    ):
        image_ids = kwargs.pop("image_ids", None)
        assert self.means3d.shape[0] == self.opacities.shape[0]
        opacities = torch.sigmoid(self.opacities)
        colors = torch.cat([self.sh0, self.shN], 1)
        scales = torch.exp(self.scales)

        render_colors, render_alphas, info = rasterization(
            means=self.means3d,
            # means=self.means3d,
            quats=self.quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.config.packed,
            absgrad=self.config.absgrad,
            sparse_grad=self.config.sparse_grad,
            rasterize_mode="antialiased" if self.config.antialiased else "classic",
            **kwargs,
        )

        return render_colors, render_alphas, info

    def _create_optimizers(self) -> list[Optimizer]:
        params = [
            # name, value, lr
            # ("means3d", self.means3d, self.lr_means3d),
            ("scales", self.scales, self.lr_scales),
            ("opacities", self.opacities, self.lr_opacities),
        ]

        if hasattr(self, "features"):
            params.append(("features", self.features, self.lr_features))
        else:
            params.append(("sh0", self.sh0, self.lr_sh0))
            params.append(("shN", self.shN, self.lr_shN))

        params.append(("colors", self.colors, self.lr_colors))

        optimizers = [
            (SparseAdam if self.config.sparse_grad else Adam)(
                [
                    {
                        "params": param,
                        "lr": lr * math.sqrt(self.batch_size),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(self.batch_size),
                betas=(
                    1 - self.batch_size * (1 - 0.9),
                    1 - self.batch_size * (1 - 0.999),
                ),
            )
            for name, param, lr in params
        ]

        return optimizers

    @torch.no_grad()
    def update_running_stats(self, info: dict):
        """Update running stats."""

        # normalize grads to [-1, 1] screen space
        if self.config.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * self.batch_size
        grads[..., 1] *= info["height"] / 2.0 * self.batch_size
        if self.config.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.opacities, max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.state_dict()[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.scales[sel])  # [N, 3]
        quats = F.normalize(self.quats[sel], dim=-1)  # [N, 4]
        # quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=DEVICE),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=DEVICE)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.state_dict()[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros((len(sel), *v.shape[1:]), device=DEVICE)
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.state_dict()[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.state_dict()[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

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
            sh_degree=self.config.sh_degree,  # active all SH degrees
            # radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
