from datetime import datetime
from typing import Literal

import torch
from matplotlib import pyplot as plt
from torch import Tensor

import wandb
from ..my_gsplat.geometry import compute_silhouette_diff


class WandbLogger:
    def __init__(self, run_name: str | None = None, config: dict = None):
        """
        Initialize the Weights & Biases logging.
        use wandb login with api key https://wandb.ai/authorize
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            run_name = run_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project="ABGICP",
            entity="supavision",
            name=run_name,
            config=config,
        )

        print(f"Run name: {run_name}:config: {config}")

    def log_translation_error(self, eT: float, step: int):
        """
        Log the translation error to wandb.
        """
        wandb.log({"Translation Error": eT}, step=step)

    def log_rotation_error(self, eR: float, step: int):
        """
        Log the rotation error to wandb.
        """
        wandb.log({"Rotation Error": eR}, step=step)

    def log_rmse_pcd(self, rmse: float, step: int):
        """
        Log the point cloud RMSE to wandb.
        """
        wandb.log({"Point Cloud RMSE": rmse}, step=step)

    def log_com_diff(self, com_diff: float, step: int):
        """
        Log the difference in center of mass between two point clouds to wandb.
        """
        wandb.log({"COM Difference": com_diff}, step=step)

    def log_LR(self, model: torch.nn.Module, schedulers: list, step: int):
        """
        Log the learning rates of all optimizers and their scheduler types.
        :param model: The model containing the optimizers
        :param schedulers: List of schedulers corresponding to the optimizers
        :param step: Current step number
        """
        for i, (optimizer, scheduler) in enumerate(zip(model.optimizers, schedulers)):
            for j, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                param_name = param_group.get("name", f"optimizer_{i}_group_{j}")
                s_type = f"Sch:{scheduler.__class__.__name__}"
                l_name = s_type + f" LR: {param_name}"
                wandb.log({l_name: lr}, step=step)

    # BUG: failed to show in wandb
    def log_gradients(
        self,
        model: torch.nn.Module,
        idx: int,
        log: Literal["gradients", "parameters", "all"] | None = "gradients",
        log_graph: bool = True,
    ):
        """https://docs.wandb.ai/ref/python/watch"""
        wandb.watch(model, log=log, log_graph=log_graph, idx=idx, log_freq=1)

    def log_iter_times(self, iter_times: int, step: int):
        """
        Log the iteration times to wandb.
        """
        wandb.log({"Iteration Times": iter_times}, step=step)

    def log_align_error(self, align_error: float, step: int):
        """
        Log the alignment error to wandb.
        """
        wandb.log({"Alignment Error": align_error}, step=step)

    def log_loss(self, name: str, loss_val: float, step: int, l_type: str = ""):
        """
        Log the loss to wandb.
        """
        wandb.log({f"{name}_{l_type}": loss_val}, step=step)

    def finish(self):
        """
        Finish the wandb run.
        """
        wandb.finish()

    def plot_rgbd(
        self,
        depth: Tensor,
        rastered_depth: Tensor,
        depth_loss: dict,
        step: int,
        *,
        color: Tensor | None = None,
        rastered_color: Tensor | None = None,
        color_loss: dict | None = None,
        silhouette_loss: dict | None = None,
        fig_title="RGBD Visualization",
    ):
        # Ensure depth tensors have a batch dimension
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)  # Reshape [H, W] to [1, H, W]
        if rastered_depth.dim() == 2:
            rastered_depth = rastered_depth.unsqueeze(0)

        silhouette_diff = compute_silhouette_diff(depth, rastered_depth)

        # Determine Plot Aspect Ratio
        aspect_ratio = depth.shape[2] / depth.shape[1]
        fig_height = 8
        fig_width = aspect_ratio * 14 / 1.55
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))

        if color is not None:
            if color.dim() == 3 and color.shape[1] == 3:  # (H, C, W)
                color = color.permute(1, 2, 0)  # -> (C, W, H)
            if color.dim() == 3:
                color = color.unsqueeze(0)  # -> B, C, W, H

            # (B, H, W, C)
            if color.shape[1] == 3:
                color = color.permute(0, 2, 3, 1)

            axs[0, 0].imshow(color[0].detach().cpu())
            axs[0, 0].set_title(
                f"Ground Truth RGB\n{color_loss['type']}: {color_loss['value']:.4f}"
            )
        else:
            axs[0, 0].set_visible(False)  # 如果没有提供彩色图像则隐藏

        axs[0, 1].imshow(depth.squeeze().detach().cpu(), cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title(
            f"Ground Truth Depth\n{depth_loss['type']}: {depth_loss['value']:.4f}"
        )

        axs[0, 2].imshow(silhouette_diff.detach().cpu(), cmap="gray")
        if silhouette_loss is not None:
            axs[0, 2].set_title(
                f"Silhouette Diff\n {silhouette_loss['type']}: {silhouette_loss['value']:.4f} "
            )

        if rastered_color is not None:
            if rastered_color.dim() == 3 and rastered_color.shape[1] == 3:  # (H, C, W)
                rastered_color = rastered_color.permute(1, 2, 0)  # -> (C, W, H)
            if rastered_color.dim() == 3:
                rastered_color = rastered_color.unsqueeze(0)  # -> B, C, W, H

            # (B, H, W, C)
            if rastered_color.shape[1] == 3:
                rastered_color = rastered_color.permute(0, 2, 3, 1)
            axs[1, 0].imshow(rastered_color[0].detach().cpu())
            axs[1, 0].set_title(
                f"Rasterized RGB\n{color_loss['type']}: {color_loss['value']:.4f}"
            )
        else:
            axs[1, 0].set_visible(False)  # 如果没有提供重建彩色图像则隐藏

        axs[1, 1].imshow(
            rastered_depth.squeeze().detach().cpu(), cmap="jet", vmin=0, vmax=6
        )
        axs[1, 1].set_title(
            f"Rasterized Depth\n{depth_loss['type']}: {depth_loss['value']:.4f}"
        )

        # Calculate depth difference and display
        diff_depth = torch.abs(depth - rastered_depth).detach().cpu()
        axs[1, 2].imshow(diff_depth.squeeze(), cmap="jet", vmin=0, vmax=6)
        axs[1, 2].set_title("Diff Depth L1")

        for ax in axs.flatten():
            if ax.get_visible():
                ax.axis("off")

        fig.suptitle(fig_title, y=1, fontsize=16)
        fig.tight_layout()

        wandb.log({fig_title: wandb.Image(fig)}, step=step)

        plt.close()
