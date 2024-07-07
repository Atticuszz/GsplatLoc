import json
import time
from timeit import default_timer

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor

from src.eval.experiment import ExperimentBase, WandbConfig
from src.my_gsplat.datasets.normalize import transform_points

from .datasets.base import Config
from .datasets.dataset import AlignData, Parser
from .loss import compute_silhouette_loss
from .model import (
    CameraOptModule_6d_inc_tans_inc,
    CameraOptModule_6d_tans,
    CameraOptModule_quat_tans,
    GSModel,
)
from .utils import (
    DEVICE,
    calculate_rotation_error,
    calculate_translation_error,
    set_random_seed,
)


class Runner(ExperimentBase):
    """Engine for training and testing."""

    def __init__(
        self,
        wandb_config: WandbConfig,
        base_config: Config = Config(),
        extra_config: dict = None,
    ) -> None:
        super().__init__(wandb_config=wandb_config, extra_config=extra_config)
        set_random_seed(42)
        # Setup output directories.

        self.config = base_config
        self.config.make_dir()
        # load data
        self.parser = Parser(self.sub_set, normalize=wandb_config.normalize)

        cam_opt_dict = {
            "quat": CameraOptModule_quat_tans,
            "6d": CameraOptModule_6d_tans,
            "6d+": CameraOptModule_6d_inc_tans_inc,
        }
        self.camera_opt_module = cam_opt_dict[extra_config["cam_opt"]]
        # Losses & Metrics.
        self.config.init_loss()

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        # Dump self.
        # with open(f"{self.config.res_dir.as_posix()}/self.json", "w") as f:
        #     json.dump(vars(self), f, cls=CustomEncoder)

        for i, train_data in enumerate([self.parser[924]]):
            # NOTE: train data loop
            train_data: AlignData
            height, width = train_data.pixels.shape[:2]
            Ks = self.parser.K.unsqueeze(0)  # [1, 3, 3]

            max_steps = self.config.max_steps
            # NOTE: Models init with tar.points
            gs_splats = GSModel(
                train_data.tar_points, train_data.colors[: train_data.tar_nums]
            ).to(DEVICE)
            # NOTE: un-project to get depth_gt for normed src_pcd
            src_points_cam = transform_points(
                torch.linalg.inv(train_data.tar_c2w),
                train_data.points[train_data.tar_nums :],
            )
            if self.parser.normalize:
                unproject_depth = GSModel(
                    src_points_cam,
                    train_data.colors[train_data.tar_nums :],
                )
                unprojected, _, _ = unproject_depth(
                    camtoworlds=torch.eye(4, device=DEVICE).unsqueeze(0),  # [1, 4, 4],
                    Ks=Ks,
                    width=width,
                    height=height,
                    # sh_degree=sh_degree_to_use,
                )
                if unprojected.shape[-1] == 4:
                    _, depths_gt = unprojected[..., 0:3], unprojected[..., 3:4]
                else:
                    _, depths_gt = unprojected, None
            else:
                depths_gt = train_data.src_depth.unsqueeze(-1).unsqueeze(0)

            # camera init with tar.pose
            camera_opt = self.camera_opt_module(train_data.tar_c2w).to(DEVICE)

            schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(
                    camera_opt.optimizers[0], gamma=0.2 ** (1.0 / max_steps)
                ),
                torch.optim.lr_scheduler.ExponentialLR(
                    camera_opt.optimizers[1], gamma=0.2 ** (1.0 / max_steps)
                ),
            ]

            # nerf viewer
            if not self.config.disable_viewer:
                self.config.init_view(gs_splats.viewer_render_fn)
            init_step = 0
            pbar = tqdm.tqdm(range(init_step, max_steps))
            for step in pbar:
                # NOTE: Training loop.
                global_tic = default_timer()
                if not self.config.disable_viewer:
                    while self.config.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.config.viewer.lock.acquire()
                    tic = default_timer()

                # NOTE: start forward
                pixels = train_data.pixels.unsqueeze(0)  # [1, H, W, 3]
                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                # # # sh schedule
                # sh_degree_to_use = min(step // self.sh_degree_interval, self.sh_degree)
                # NOTE: apply c2w to src_gs
                # transformed_points, cur_c2w = camera_opt(train_data.src_points)
                cur_c2w = camera_opt()
                # gs_splats.means3d = torch.cat(
                #     [train_data.tar_points, transformed_points], dim=0
                # )
                # NOTE: gs forward
                renders, alphas, info = gs_splats(
                    camtoworlds=cur_c2w.unsqueeze(0),  # [1, 3, 3],
                    Ks=Ks,  # [1, 3, 3],
                    width=width,
                    height=height,
                    # sh_degree=sh_degree_to_use,
                )

                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None
                info["means2d"].retain_grad()  # used for running stats
                # NOTE:loss
                # avoid nan area
                non_zero_depth_mask = (depths != 0).float()

                # RGB L1 Loss
                l1loss = F.l1_loss(
                    colors * non_zero_depth_mask,
                    pixels * non_zero_depth_mask,
                    reduction="sum",
                ) / (non_zero_depth_mask.sum() + 1e-8)

                # SSIM Loss
                ssim_value = self.config.ssim(
                    (pixels * non_zero_depth_mask).permute(0, 3, 1, 2),
                    (colors * non_zero_depth_mask).permute(0, 3, 1, 2),
                )
                ssimloss = 1.0 - ssim_value

                # Combined RGB Loss
                rgb_loss = (
                    l1loss * (1.0 - self.config.ssim_lambda)
                    + ssimloss * self.config.ssim_lambda
                ) * (1 - 1)

                # Depth Loss
                depth_loss = F.l1_loss(
                    depths * non_zero_depth_mask,
                    depths_gt * non_zero_depth_mask,
                    reduction="sum",
                ) / (non_zero_depth_mask.sum() + 1e-8)
                depth_loss *= train_data.scene_scale

                # Silhouette Loss
                silhouette_loss = compute_silhouette_loss(
                    (
                        depths.squeeze(-1).squeeze(0)
                        * non_zero_depth_mask.squeeze(-1).squeeze(0)
                    ),
                    (
                        depths_gt.squeeze(-1).squeeze(0)
                        * non_zero_depth_mask.squeeze(-1).squeeze(0)
                    ),
                    loss_type="l1",
                )

                # Total Loss
                total_loss = (
                    rgb_loss
                    + depth_loss * self.config.depth_lambda
                    + silhouette_loss * (1 - self.config.depth_lambda)
                )

                total_loss.backward(retain_graph=True)
                # NOTE: logger
                with torch.no_grad():
                    # loss
                    self.logger.log_loss("total_loss", total_loss.item(), step=step)
                    # self.logger.log_loss(
                    #     "pixels", l1loss.item(), step=step, l_type="l1"
                    # )
                    # self.logger.log_loss(
                    #     "pixels", ssimloss.item(), step=step, l_type="ssim"
                    # )
                    self.logger.log_loss(
                        "depth", depth_loss.item(), step=step, l_type="l1"
                    )
                    self.logger.log_loss(
                        "silhouette_loss",
                        silhouette_loss.item(),
                        step=step,
                        l_type="l1",
                    )
                    # IMAGE
                    if step % 50 == 0:
                        self.logger.plot_rgbd(
                            depths_gt.squeeze(-1).squeeze(0),
                            depths.squeeze(-1).squeeze(0),
                            # combined_projected_depth,
                            {
                                "type": "l1",
                                "value": depth_loss.item(),
                            },
                            color=train_data.pixels,
                            rastered_color=colors.squeeze(0),
                            color_loss={
                                "type": "pixels_l1",
                                "value": l1loss.item(),
                            },
                            step=step,
                            fig_title="gs_splats Visualization",
                        )
                    # NOTE: monitor the pose error
                    eT = calculate_translation_error(
                        cur_c2w.squeeze(-1).squeeze(0),
                        train_data.src_c2w.squeeze(-1).squeeze(0),
                    )

                    eR = calculate_rotation_error(
                        cur_c2w.squeeze(-1).squeeze(0),
                        train_data.src_c2w.squeeze(-1).squeeze(0),
                    )
                    # Error
                    self.logger.log_translation_error(eT, step=step)
                    self.logger.log_rotation_error(eR, step=step)
                    # LR
                    self.logger.log_LR(
                        model=camera_opt,
                        schedulers=schedulers,
                        step=step,
                    )
                    # self.logger.log_LR(
                    #     model=gs_splats,
                    #     schedulers=schedulers[2:],
                    #     step=step,
                    # )

                    # NOTE: early stop
                    desc = f"loss={total_loss.item():.8f}|"

                    pbar.set_description(desc)
                    if self.config.early_stop:

                        if eR < self.config.best_eR and eT < self.config.best_eT:
                            self.config.best_eR = eR
                            self.config.best_eT = eT
                            self.config.counter = 0  # 重置计数器
                        else:
                            self.config.counter += 1
                        desc += f"best_eR:{self.config.best_eR}| best_eT: {self.config.best_eT}|"
                        if self.config.counter >= self.config.patience:
                            desc += f"\nEarly stopping triggered at step {step}|"
                            pbar.set_description(desc)
                            self.config.counter = 0
                            break

                # NOTE: early stop
                desc = f"loss={total_loss.item():.8f}|"
                desc += (
                    f"best_eR:{self.config.best_eR}| best_eT: {self.config.best_eT}|"
                )
                pbar.set_description(desc)
                if self.config.early_stop:
                    if eR < self.config.best_eR and eT < self.config.best_eT:
                        self.config.best_eR = eR
                        self.config.best_eT = eT
                        self.config.counter = 0  # 重置计数器
                    else:
                        self.config.counter += 1
                    if self.config.counter >= self.config.patience:
                        desc += f"\nEarly stopping triggered at step {step}|"
                        pbar.set_description(desc)
                        self.config.counter = 0
                        break

                # # NOTE: update running stats for prunning & growing
                # if step < self.refine_stop_iter:
                #     gs_splats.update_running_stats(info)
                #
                #     if step > self.refine_start_iter and step % self.refine_every == 0:
                #         grads = gs_splats.running_stats[
                #             "grad2d"
                #         ] / gs_splats.running_stats["count"].clamp_min(1)
                #
                #         # grow GSs
                #         is_grad_high = grads >= self.grow_grad2d
                #         is_small = (
                #             torch.exp(gs_splats.scales).max(dim=-1).values
                #             <= self.grow_scale3d * train_data.scene_scale
                #         )
                #         is_dupli = is_grad_high & is_small
                #         n_dupli = is_dupli.sum().item()
                #         gs_splats.refine_duplicate(is_dupli)
                #
                #         is_split = is_grad_high & ~is_small
                #         is_split = torch.cat(
                #             [
                #                 is_split,
                #                 # new GSs added by duplication will not be split
                #                 torch.zeros(n_dupli, device=DEVICE, dtype=torch.bool),
                #             ]
                #         )
                #         n_split = is_split.sum().item()
                #         gs_splats.refine_split(is_split)
                #         print(
                #             f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                #             f"Now having {len(gs_splats)} GSs."
                #         )
                #
                #         # prune GSs
                #         is_prune = torch.sigmoid(gs_splats.opacities) < self.prune_opa
                #         if step > self.reset_every:
                #             # The official code also implements sreen-size pruning but
                #             # it's actually not being used due to a bug:
                #             # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                #             is_too_big = (
                #                 torch.exp(gs_splats.scales).max(dim=-1).values
                #                 > self.prune_scale3d * self.scene_scale
                #             )
                #             is_prune = is_prune | is_too_big
                #         n_prune = is_prune.sum().item()
                #         gs_splats.refine_keep(~is_prune)
                #         print(
                #             f"Step {step}: {n_prune} GSs pruned. "
                #             f"Now having {len(gs_splats)} GSs."
                #         )
                #
                #         # reset running stats
                #         gs_splats.running_stats["grad2d"].zero_()
                #         gs_splats.running_stats["count"].zero_()
                #
                #     if step % self.reset_every == 0:
                #         gs_splats.reset_opa(self.prune_opa * 2.0)
                #
                # # Turn Gradients into Sparse Tensor before running optimizer
                # if self.sparse_grad:
                #     assert self.packed, "Sparse gradients only work with packed mode."
                #     gaussian_ids = info["gaussian_ids"]
                #     for k in gs_splats.keys():
                #         grad = gs_splats[k].grad
                #         if grad is None or grad.is_sparse:
                #             continue
                #         gs_splats[k].grad = torch.sparse_coo_tensor(
                #             indices=gaussian_ids[None],  # [1, nnz]
                #             values=grad[gaussian_ids],  # [nnz, ...]
                #             size=gs_splats[k].size(),  # [N, ...]
                #             is_coalesced=len(Ks) == 1,
                #         )

                # optimize
                # for optimizer in gs_splats.optimizers:
                #     optimizer.step()
                #     optimizer.zero_grad(set_to_none=True)
                for optimizer in camera_opt.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()

                # # save checkpoint
                # if step in [i - 1 for i in self.save_steps] or step == max_steps - 1:
                #     mem = torch.cuda.max_memory_allocated() / 1024**3
                #     stats = {
                #         "mem": mem,
                #         "ellipse_time": time.time() - global_tic,
                #         "num_GS": len(gs_splats),
                #     }
                #     print("Step: ", step, stats)
                #     with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                #         json.dump(stats, f)
                #     torch.save(
                #         {
                #             "step": step,
                #             "splats": gs_splats.state_dict(),
                #         },
                #         f"{self.ckpt_dir}/ckpt_{step}.pt",
                #     )

                # if step in [i - 1 for i in self.eval_steps] or step == max_steps - 1:
                #     self.eval(gs_splats, cur_c2w, step)

                # viewer
                if not self.config.disable_viewer:
                    self.config.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.config.viewer.state.num_train_rays_per_sec = (
                        num_train_rays_per_sec
                    )
                    # Update the scene.
                    self.config.viewer.update(step, num_train_rays_per_step)
            self.logger.finish()

    @torch.no_grad()
    def eval(self, gs_splats: GSModel, c2w: Tensor, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")

        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        Ks = self.parser.K.unsqueeze(0)
        pixels = self.parser[step].pixels.unsqueeze(0) / 255.0
        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()
        tic = time.time()

        colors, _, _ = gs_splats(
            camtoworlds=c2w.unsqueeze(0),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )  # [1, H, W, 3]
        colors = torch.clamp(colors, 0.0, 1.0)
        torch.cuda.synchronize()
        ellipse_time += time.time() - tic

        # write images
        canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
        imageio.imwrite(
            f"{self.render_dir}/val_{step:04d}.png", (canvas * 255).astype(np.uint8)
        )

        pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        metrics["psnr"].append(self.psnr(colors, pixels))
        metrics["ssim"].append(self.ssim(colors, pixels))
        metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(self.parser)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(gs_splats)}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(gs_splats),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()
