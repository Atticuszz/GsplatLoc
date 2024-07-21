import time
from timeit import default_timer

import torch
import torch.nn.functional as F
import tqdm

from src.eval.experiment import ExperimentBase, WandbConfig

from .datasets.base import Config
from .datasets.dataset import AlignData, Parser
from .loss import compute_depth_loss, compute_silhouette_loss
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
        torch.set_float32_matmul_precision("high")

        for i, train_data in enumerate(self.parser):
            # NOTE: train data loop
            train_data: AlignData
            height, width = train_data.pixels.shape[:2]
            Ks = self.parser.K.unsqueeze(0)  # [1, 3, 3]

            max_steps = self.config.max_steps
            # NOTE: Models init with tar.points
            gs_splats = GSModel(
                train_data.tar_points, train_data.colors[: train_data.tar_nums]
            ).to(DEVICE)

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
                # NOTE: apply c2w to src_gs
                cur_c2w = camera_opt()
                # NOTE: gs forward
                renders, alphas, info = gs_splats(
                    camtoworlds=cur_c2w.unsqueeze(0),  # [1, 3, 3],
                    Ks=Ks,  # [1, 3, 3],
                    width=width,
                    height=height,
                )

                assert renders.shape[-1] == 4
                colors, _depths = renders[..., 0:3], renders[..., 3:4]

                # scale depth after normalized
                if self.parser.normalize:
                    depths = _depths / train_data.scale_factor
                else:
                    depths = _depths

                # NOTE:loss
                # avoid nan area
                non_zero_depth_mask = (depths != 0).float()

                # # RGB L1 Loss
                # l1loss = F.l1_loss(
                #     colors * non_zero_depth_mask,
                #     pixels * non_zero_depth_mask,
                #     reduction="sum",
                # ) / (non_zero_depth_mask.sum() + 1e-8)

                # # SSIM Loss
                # ssim_value = self.config.ssim(
                #     (pixels * non_zero_depth_mask).permute(0, 3, 1, 2),
                #     (colors * non_zero_depth_mask).permute(0, 3, 1, 2),
                # )
                # ssimloss = 1.0 - ssim_value

                # Depth Loss
                depth_loss = compute_depth_loss(
                    depths * non_zero_depth_mask,
                    depths_gt * non_zero_depth_mask,
                    loss_type="l1",
                )

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
                total_loss = depth_loss * self.config.depth_lambda + silhouette_loss * (
                    1 - self.config.depth_lambda
                )

                total_loss.backward(retain_graph=True)
                # NOTE: logger
                with torch.no_grad():

                    # NOTE: early stop
                    desc = f"loss={total_loss.item():.8f}|"
                    pbar.set_description(desc)

                    if self.config.early_stop:
                        # NOTE: monitor the pose error
                        eT = calculate_translation_error(
                            cur_c2w.squeeze(-1).squeeze(0),
                            train_data.src_c2w.squeeze(-1).squeeze(0),
                        )

                        eR = calculate_rotation_error(
                            cur_c2w.squeeze(-1).squeeze(0),
                            train_data.src_c2w.squeeze(-1).squeeze(0),
                        )
                        if depth_loss.item() < self.config.best_loss:
                            self.config.best_loss = depth_loss.item()
                            self.config.best_eT = eT
                            self.config.best_eR = eR
                            self.config.counter = 0
                        else:
                            self.config.counter += 1
                        desc += f"best_eR:{self.config.best_eR}| best_eT: {self.config.best_eT}|cur_i:{i}|"
                        pbar.set_description(desc)
                        if self.config.counter >= self.config.patience:
                            desc += f"Early stopping triggered at step {step}|"
                            # clean
                            (
                                self.config.counter,
                                self.config.best_loss,
                                self.config.best_eT,
                                self.config.best_eR,
                            ) = (0, float("inf"), float("inf"), float("inf"))

                            pbar.set_description(desc)

                            # NOTE: log here
                            # loss
                            self.logger.log_loss(
                                "total_loss", total_loss.item(), step=i
                            )
                            # self.logger.log_loss(
                            #     "pixels", l1loss.item(), step=i, l_type="l1"
                            # )
                            # self.logger.log_loss(
                            #     "pixels", ssimloss.item(), step=i, l_type="ssim"
                            # )
                            self.logger.log_loss(
                                "depth", depth_loss.item(), step=i, l_type="l1"
                            )
                            self.logger.log_loss(
                                "silhouette_loss",
                                silhouette_loss.item(),
                                step=i,
                                l_type="l1",
                            )
                            # NOTE: IMAGE
                            #
                            # psnr = self.config.psnr(
                            #     (pixels * non_zero_depth_mask).permute(0, 3, 1, 2),
                            #     (colors * non_zero_depth_mask).permute(0, 3, 1, 2),
                            # )
                            self.logger.plot_rgbd(
                                depths_gt.squeeze(-1).squeeze(0),
                                depths.squeeze(-1).squeeze(0),
                                # combined_projected_depth,
                                {
                                    "type": "l1",
                                    "value": depth_loss.item(),
                                },
                                # color=train_data.pixels,
                                # rastered_color=colors.squeeze(0),
                                # color_loss={
                                #     "type": "psnr",
                                #     "value": psnr.item(),
                                # },
                                step=i,
                                fig_title="gs_splats Visualization",
                            )

                            # Error
                            self.logger.log_translation_error(eT, step=i)
                            self.logger.log_rotation_error(eR, step=i)
                            # LR
                            self.logger.log_LR(
                                model=camera_opt,
                                schedulers=schedulers,
                                step=i,
                            )
                            break

                for optimizer in camera_opt.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()

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
