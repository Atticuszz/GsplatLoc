import math
import os
import time

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import viser
from gsplat._helper import load_test_data
from gsplat.rendering import rasterization


class GSViewer:
    def __init__(
        self,
        output_dir: str = "results/",
        scene_grid: int = 1,
        ckpt: str = None,
        port: int = 8080,
        backend: str = "gsplat",
    ):
        self.output_dir = output_dir
        self.scene_grid = scene_grid
        self.ckpt = ckpt
        self.port = port
        self.backend = backend

        assert self.scene_grid % 2 == 1, "scene_grid must be odd"
        torch.manual_seed(42)
        self.device = "cuda"

        self.load_data()

    def load_data(self):
        if self.ckpt is None:
            (
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                self.viewmats,
                self.Ks,
                self.width,
                self.height,
            ) = load_test_data(device=self.device, scene_grid=self.scene_grid)
            self.sh_degree = None
            self.C = len(self.viewmats)
            self.N = len(self.means)
            print("Number of Gaussians:", self.N)
        else:
            ckpt = torch.load(self.ckpt, map_location=self.device)["splats"]
            self.means = ckpt["means3d"]
            self.quats = F.normalize(ckpt["quats"], p=2, dim=-1)
            self.scales = torch.exp(ckpt["scales"])
            self.opacities = torch.sigmoid(ckpt["opacities"])
            sh0 = ckpt["sh0"]
            shN = ckpt["shN"]
            self.colors = torch.cat([sh0, shN], dim=-2)
            self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)

            # crop
            aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=self.device)
            edges = aabb[3:] - aabb[:3]
            sel = ((self.means >= aabb[:3]) & (self.means <= aabb[3:])).all(dim=-1)
            sel = torch.where(sel)[0]
            self.means, self.quats, self.scales, self.colors, self.opacities = (
                self.means[sel],
                self.quats[sel],
                self.scales[sel],
                self.colors[sel],
                self.opacities[sel],
            )

            # repeat the scene into a grid (to mimic a large-scale setting)
            repeats = self.scene_grid
            gridx, gridy = torch.meshgrid(
                [
                    torch.arange(-(repeats // 2), repeats // 2 + 1, device=self.device),
                    torch.arange(-(repeats // 2), repeats // 2 + 1, device=self.device),
                ],
                indexing="ij",
            )
            grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
                -1, 3
            )
            self.means = (
                self.means[None, :, :] + grid[:, None, :] * edges[None, None, :]
            )
            self.means = self.means.reshape(-1, 3)
            self.quats = self.quats.repeat(repeats**2, 1)
            self.scales = self.scales.repeat(repeats**2, 1)
            self.colors = self.colors.repeat(repeats**2, 1, 1)
            self.opacities = self.opacities.repeat(repeats**2)

    def render_batch(self):
        if self.ckpt is None:
            render_colors, render_alphas, meta = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                self.viewmats,
                self.Ks,
                self.width,
                self.height,
                render_mode="RGB+D",
            )
            assert render_colors.shape == (self.C, self.height, self.width, 4)
            assert render_alphas.shape == (self.C, self.height, self.width, 1)

            render_rgbs = render_colors[..., 0:3]
            render_depths = render_colors[..., 3:4]
            render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(self.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                    [
                        render_rgbs.reshape(self.C * self.height, self.width, 3),
                        render_depths.reshape(
                            self.C * self.height, self.width, 1
                        ).expand(-1, -1, 3),
                        render_alphas.reshape(
                            self.C * self.height, self.width, 1
                        ).expand(-1, -1, 3),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{self.output_dir}/render.png", (canvas * 255).astype(np.uint8)
            )
            print(canvas)
        else:
            print("Batch rendering is not supported for custom checkpoints.")

    @torch.no_grad()
    def viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: tuple[int, int]
    ):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        viewmat = c2w.inverse()

        if self.backend == "gsplat":
            rasterization_fn = rasterization
        elif self.backend == "gsplat_legacy":
            from gsplat import rasterization_legacy_wrapper

            rasterization_fn = rasterization_legacy_wrapper
        elif self.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError("Invalid backend")
        render_colors, render_alphas, meta = rasterization_fn(
            self.means,  # [N, 3]
            self.quats,  # [N, 4]
            self.scales,  # [N, 3]
            self.opacities,  # [N]
            self.colors,  # [N, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=self.sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            # radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    def run_viewer(self):
        server = viser.ViserServer(port=self.port, verbose=False)
        _ = nerfview.Viewer(
            server=server,
            render_fn=self.viewer_render_fn,
            mode="rendering",
        )
        print(f"Viewer running on port {self.port}... Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Viewer stopped.")


if __name__ == "__main__":
    viewer = GSViewer(
        output_dir="results/",
        scene_grid=1,
        ckpt="/home/atticuszz/DevSpace/python/AB_GICP/src/results/Replica/ckpts/ckpt_199.pt",  # Set to None for test data, or specify your checkpoint file
        port=8080,
        backend="gsplat",
    )
    viewer.render_batch()
    viewer.run_viewer()
