import torch
from torch import Tensor, nn

from src.eval.logger import WandbLogger
from src.pose_estimation.gemoetry import (
    construct_full_pose,
    project_depth,
    unproject_depth,
    normalize_depth,
    unnormalize_depth,
)
from src.pose_estimation.loss import compute_depth_loss, compute_silhouette_loss


class PoseEstimationModel(nn.Module):
    """
    Initialize the CameraPoseEstimator with camera intrinsics and computation device.

    Parameters
    ----------
    intrinsics : torch.Tensor
        Intrinsic camera matrix with dimensions [3, 3].
    device : str, optional
        The device on which to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
    """

    def __init__(
        self,
        intrinsics: Tensor,
        init_pose: Tensor,
        device="cuda",
        logger: WandbLogger | None = None,
    ):
        super().__init__()
        self.intrinsics = intrinsics
        self.rotation_last = init_pose[:3, :3]
        self.translation_last = init_pose[:3, 3]
        self.rotation_cur = nn.Parameter(init_pose[:3, :3])
        self.translation_cur = nn.Parameter(init_pose[:3, 3])

        # self.pose_last = init_pose
        # self.pose_cur = nn.Parameter(init_pose)
        self.device = device

        self.logger = logger

    def forward(self, depth_last, depth_current, i: int | None):
        """
        Parameters
        ----------
        depth_last : torch.Tensor
            The depth image from the previous frame with dimensions [height, width].
        depth_current : torch.Tensor
            The depth image from the current frame with dimensions [height, width].
        Returns
        -------
        loss: torch.Tensor
            The loss value.
        """
        rotation_last = self.rotation_last
        translation_last = self.translation_last
        pose_last = construct_full_pose(rotation_last, translation_last)
        pose_cur = construct_full_pose(self.rotation_cur, self.translation_cur)

        # Normalize depth images
        normalized_depth_last, min_last, max_last = normalize_depth(depth_last)
        normalized_depth_current, min_current, max_current = normalize_depth(
            depth_current
        )

        # depth to pcd
        pcd_last = project_depth(normalized_depth_last, pose_last, self.intrinsics)
        pcd_current = project_depth(normalized_depth_current, pose_cur, self.intrinsics)

        # projected to depth
        projected_depth_last = unproject_depth(pcd_last, pose_cur, self.intrinsics)
        projected_depth_current = unproject_depth(
            pcd_current, pose_cur, self.intrinsics
        )

        # combined
        combined_projected_depth = torch.min(
            projected_depth_last, projected_depth_current
        )
        combined_projected_depth[combined_projected_depth == 0] = torch.max(
            projected_depth_last, projected_depth_current
        )[combined_projected_depth == 0]

        # NOTE: Calculate depth  loss
        depth_loss = compute_depth_loss(
            combined_projected_depth,
            depth_current,
            loss_type="InverseDepthSmoothnessLoss",
        )

        # # NOTE:  silhouette  loss
        silhouette_loss = compute_silhouette_loss(
            combined_projected_depth, depth_current, loss_type="mse"
        )

        # # # Regularization term: penalize large changes in the pose
        # reg_loss = torch.sum((self.pose_cur - self.pose_last) ** 2)

        # NOTE: Regularization for pose changes
        reg_loss_rotation = torch.sum((self.rotation_cur - rotation_last) ** 2)
        reg_loss_translation = torch.sum((self.translation_cur - translation_last) ** 2)

        # NOTE: Combine losses
        total_loss = (
            0.1 * depth_loss
            + 0.1 * (reg_loss_rotation + reg_loss_translation)
            + 0.8 * silhouette_loss
        )
        if self.logger is not None:
            # loss
            self.logger.log_loss("total_loss", total_loss.item(), i)
            self.logger.log_loss("depth_loss_mse", depth_loss.item(), i)
            self.logger.log_loss("reg_loss_rotation", reg_loss_rotation.item(), i)
            self.logger.log_loss("reg_loss_translation", reg_loss_translation.item(), i)
            self.logger.log_loss("silhouette_loss_mse", silhouette_loss.item(), i)
            # log image
            self.logger.plot_rgbd(
                depth_current,
                unnormalize_depth(combined_projected_depth, min_current, max_current),
                {"type": "InverseDepthSmoothnessLoss", "value": depth_loss.item()},
                i,
                silhouette_loss={"type": "mse", "value": silhouette_loss.item()},
                fig_title="RGBD Visualization",
            )
        return total_loss
