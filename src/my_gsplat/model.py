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


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


@dataclass
class CameraConfig:
    pose_opt: bool = True
    trans_lr: float = 1e-3
    quat_lr: float = 1e-3
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0


class CameraOptModule(nn.Module, CameraConfig):
    def __init__(self, init_pose: Tensor):
        super().__init__()
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
            ("quat", self.quaternion_cur, self.quat_lr),
            ("trans", self.translation_cur, self.trans_lr),
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
                weight_decay=self.pose_opt_reg,
            )
            for name, param, lr in params
        ]
        return optimizers


@dataclass
class GsConfig:
    ssim_lambda: float = 0.2
    init_opa: float = 1.0
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


class GSModel(nn.Module, GsConfig):
    def __init__(
        self,
        # dataset
        gs_data: AlignData,
        # config
        batch_size: int = 1,
        feature_dim: int | None = None,
    ):
        super().__init__()
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
            torch.logit(torch.full((points.shape[0],), self.init_opa, device=DEVICE))
        )  # [N,]
        # NOTE: no deformation
        self.quats = torch.nn.Parameter(
            to_tensor([1, 0, 0, 0], requires_grad=True).repeat(points.shape[0], 1)
        )  # [N, 4]
        # self.quats = torch.nn.Parameter(quats)

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros(
                (points.shape[0], (self.sh_degree + 1) ** 2, 3), device=DEVICE
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
            packed=self.packed,
            absgrad=self.absgrad,
            sparse_grad=self.sparse_grad,
            rasterize_mode="antialiased" if self.antialiased else "classic",
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
            (SparseAdam if self.sparse_grad else Adam)(
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
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * self.batch_size
        grads[..., 1] *= info["height"] / 2.0 * self.batch_size
        if self.packed:
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
            sh_degree=self.sh_degree,  # active all SH degrees
            # radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
