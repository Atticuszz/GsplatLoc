from datetime import datetime

import torch
import wandb
from matplotlib import pyplot as plt


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

    def log_align_fps(self, fps: float, step: int):
        wandb.log({"Alignment Fps": fps}, step=step)

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

    def log_loss(self, loss: float, step: int):
        """
        Log the loss to wandb.
        """
        wandb.log({"Loss": loss}, step=step)

    def finish(self):
        """
        Finish the wandb run.
        """
        wandb.finish()

    def plot_rgbd_silhouette(
        self,
        color,
        depth,
        rastered_color,
        rastered_depth,
        presence_sil_mask,
        diff_depth_l1,
        psnr,
        depth_l1,
        fig_title,
        plot_name=None,
        save_plot=False,
        diff_rgb=None,
        step=None,
    ):
        # Determine Plot Aspect Ratio
        aspect_ratio = color.shape[2] / color.shape[1]
        fig_height = 8
        fig_width = 14 / 1.55
        fig_width = fig_width * aspect_ratio
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
        axs[0, 0].set_title("Ground Truth RGB")
        axs[0, 1].imshow(depth[0, :, :].cpu(), cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title("Ground Truth Depth")
        rastered_color = torch.clamp(rastered_color, 0, 1)
        axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
        axs[1, 0].set_title(f"Rasterized RGB, PSNR: {psnr:.2f}")
        axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap="jet", vmin=0, vmax=6)
        axs[1, 1].set_title(f"Rasterized Depth, L1: {depth_l1:.2f}")
        if diff_rgb is not None:
            axs[0, 2].imshow(diff_rgb.cpu(), cmap="jet", vmin=0, vmax=6)
            axs[0, 2].set_title("Diff RGB L1")
        else:
            axs[0, 2].imshow(presence_sil_mask, cmap="gray")
            axs[0, 2].set_title("Rasterized Silhouette")
        diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
        axs[1, 2].imshow(diff_depth_l1, cmap="jet", vmin=0, vmax=6)
        axs[1, 2].set_title("Diff Depth L1")
        for ax in axs.flatten():
            ax.axis("off")
        fig.suptitle(fig_title, y=0.95, fontsize=16)
        fig.tight_layout()

        if step is not None:
            wandb.log({fig_title: wandb.Image(fig)}, step=step)
        plt.close()
