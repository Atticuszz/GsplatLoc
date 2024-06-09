import torch
import torch.optim as optim
from torch import Tensor, nn

from src.eval.utils import calculate_rotation_error, calculate_translation_error
from src.pose_estimation import DEVICE
from src.pose_estimation.gemoetry import (
    construct_full_pose,
    project_depth,
    unproject_depth,
)
from src.pose_estimation.loss import depth_loss, silhouette_loss
from src.slam_data import Replica, RGBDImage
from src.utils import to_tensor


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

    def __init__(self, intrinsics: Tensor, init_pose: Tensor, device="cuda"):
        super().__init__()
        self.intrinsics = intrinsics
        self.rotation_last = init_pose[:3, :3]
        self.translation_last = init_pose[:3, 3]
        self.rotation_cur = nn.Parameter(init_pose[:3, :3])
        self.translation_cur = nn.Parameter(init_pose[:3, 3])

        # self.pose_last = init_pose
        # self.pose_cur = nn.Parameter(init_pose)
        self.device = device

    def forward(self, depth_last, depth_current):
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

        # depth to pcd
        pcd_last = project_depth(depth_last, pose_last, self.intrinsics)
        pcd_current = project_depth(depth_current, pose_cur, self.intrinsics)

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
        mse_loss = depth_loss(combined_projected_depth, depth_current)

        # NOTE:  silhouette  loss
        silhouette = silhouette_loss(combined_projected_depth, depth_current)

        # # # Regularization term: penalize large changes in the pose
        # reg_loss = torch.sum((self.pose_cur - self.pose_last) ** 2)

        # NOTE: Regularization for pose changes
        reg_loss_rotation = torch.sum((self.rotation_cur - rotation_last) ** 2)
        reg_loss_translation = torch.sum((self.translation_cur - translation_last) ** 2)

        # NOTE: Combine losses
        total_loss = (
            0.3 * mse_loss
            + 0.1 * (reg_loss_rotation + reg_loss_translation)
            + 0.6 * silhouette
        )

        return total_loss


def train_model_with_adam(
    tar_rgb_d: RGBDImage,
    src_rgb_d: RGBDImage,
    K: Tensor,
    num_iterations=50,
    learning_rate=1e-3,
) -> tuple[float, Tensor]:
    init_pose = to_tensor(tar_rgb_d.pose, DEVICE, requires_grad=True)
    model = PoseEstimationModel(K, init_pose, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # result
    min_loss = float("inf")
    best_pose = init_pose.clone()
    for i in range(num_iterations):
        optimizer.zero_grad()
        depth_last = to_tensor(tar_rgb_d.depth, DEVICE, requires_grad=True)
        depth_current = to_tensor(src_rgb_d.depth, DEVICE, requires_grad=True)
        loss = model(depth_last, depth_current)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                r, t = model.rotation_cur, model.translation_cur
                best_pose = construct_full_pose(r, t)

            if i % 10 == 0:
                print(f"Iteration {i}: Loss {min_loss}")
        # scheduler.step()
    return min_loss, best_pose


def train_model_with_LBFGS(
    tar_rgb_d: RGBDImage,
    src_rgb_d: RGBDImage,
    K: Tensor,
    num_iterations=20,
    learning_rate=1e-3,
) -> tuple[float, Tensor]:
    init_pose = to_tensor(tar_rgb_d.pose, DEVICE, requires_grad=True)
    model = PoseEstimationModel(K, init_pose, DEVICE)
    model.to(DEVICE)

    # check if all parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require gradients.")

    # 使用 LBFGS 优化器
    optimizer = optim.LBFGS(
        model.parameters(), lr=learning_rate, max_iter=1, history_size=10
    )

    def closure():
        optimizer.zero_grad()
        depth_last = to_tensor(tar_rgb_d.depth, DEVICE)
        depth_current = to_tensor(src_rgb_d.depth, DEVICE)
        loss = model(depth_last, depth_current)
        loss.backward()
        return loss

    min_loss = float("inf")
    best_pose = None

    # 执行优化
    for i in range(num_iterations):
        optimizer.step(closure)
        loss = closure()
        with torch.no_grad():
            if loss.item() < min_loss:
                min_loss = loss.item()
                r, t = model.rotation_cur.clone(), model.translation_cur.clone()
                best_pose = construct_full_pose(r, t)
            if i % 10 == 0:
                print(f"Iteration {i}: Loss {min_loss}")

    return min_loss, best_pose


def eval():
    tar_rgb_d, src_rgb_d = Replica()[2], Replica()[3]
    _, estimate_pose = train_model_with_adam(
        tar_rgb_d, src_rgb_d, to_tensor(tar_rgb_d.K, DEVICE, requires_grad=True)
    )
    # _, estimate_pose = train_model_with_LBFGS(
    #     tar_rgb_d, src_rgb_d, to_tensor(tar_rgb_d.K, DEVICE, requires_grad=True)
    # )

    eT = calculate_translation_error(
        estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose
    )
    eR = calculate_rotation_error(estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose)
    print(f"Translation error: {eT:.8f}")
    print(f"Rotation error: {eR:.8f}")


if __name__ == "__main__":
    eval()
