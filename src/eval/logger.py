from datetime import datetime

import wandb


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
