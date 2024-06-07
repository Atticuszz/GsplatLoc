import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn

from src.eval.utils import calculate_rotation_error, calculate_translation_error
from src.slam_data import Replica, RGBDImage
from src.utils import to_tensor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {DEVICE} DEVICE")


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
        self.pose_last = init_pose
        self.pose_cur = nn.Parameter(init_pose)
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
        # depth to pcd
        pcd_last = self.depth_to_pointcloud(depth_last, self.pose_last)
        pcd_current = self.depth_to_pointcloud(depth_current, self.pose_cur)

        # projected to depth
        projected_depth_last = self.pointcloud_to_depth(pcd_last, self.pose_cur)
        projected_depth_current = self.pointcloud_to_depth(pcd_current, self.pose_cur)

        # combined
        combined_projected_depth = torch.min(
            projected_depth_last, projected_depth_current
        )
        combined_projected_depth[combined_projected_depth == 0] = torch.max(
            projected_depth_last, projected_depth_current
        )[combined_projected_depth == 0]

        # Calculate MSE loss
        mse_loss = F.mse_loss(combined_projected_depth, depth_current)

        # Regularization term: penalize large changes in the pose
        reg_loss = torch.sum((self.pose_cur - self.pose_last) ** 2)

        # Combine losses
        loss = mse_loss + 0.001 * reg_loss  # Adjust regularization weight as needed
        return loss

    def depth_to_pointcloud(self, depth, pose):
        """
        Converts a depth image to a point cloud in the world coordinate system using the provided pose.

        Parameters
        ----------
        depth : torch.Tensor
            The depth image with dimensions [height, width].
        pose : torch.Tensor
            The 4x4 transformation matrix from camera to world coordinates.

        Returns
        -------
        torch.Tensor
            The converted point cloud in world coordinates with dimensions [height, width, 4].
        """
        height, width = depth.shape
        grid_x, grid_y = torch.meshgrid(
            torch.arange(width), torch.arange(height), indexing="xy"
        )
        grid_x = grid_x.float().to(self.device)
        grid_y = grid_y.float().to(self.device)

        Z = depth.to(self.device)
        X = (grid_x - self.intrinsics[0, 2]) * Z / self.intrinsics[0, 0]
        Y = (grid_y - self.intrinsics[1, 2]) * Z / self.intrinsics[1, 1]
        ones = torch.ones_like(Z)

        pcd = torch.stack((X, Y, Z, ones), dim=-1)
        pcd_world = torch.einsum("hwj,jk->hwk", pcd, pose)
        return pcd_world

    def pointcloud_to_depth(self, pcd, pose):
        """
        Projects a point cloud from world coordinates back to a depth image using the provided pose.

        Parameters
        ----------
        pcd : torch.Tensor
           The point cloud in world coordinates with dimensions [height, width, 4].
        pose : torch.Tensor
           The 4x4 transformation matrix from world to camera coordinates.

        Returns
        -------
        torch.Tensor
           The depth image created from the point cloud with dimensions [height, width].
        """
        pcd_camera = torch.einsum("hwj,jk->hwk", pcd, torch.inverse(pose))

        x = pcd_camera[..., 0]
        y = pcd_camera[..., 1]
        z = pcd_camera[..., 2]
        u = (x / z) * self.intrinsics[0, 0] + self.intrinsics[0, 2]
        v = (y / z) * self.intrinsics[1, 1] + self.intrinsics[1, 2]

        projected_depth = torch.zeros_like(z)
        height, width = z.shape
        u, v = u.long(), v.long()
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        projected_depth[v[valid], u[valid]] = z[valid]

        return projected_depth


def train_model(
    tar_rgb_d: RGBDImage,
    src_rgb_d: RGBDImage,
    K: Tensor,
    num_iterations=20,
    learning_rate=1e-6,
) -> tuple[float, Tensor]:
    init_pose = to_tensor(tar_rgb_d.pose, DEVICE)
    model = PoseEstimationModel(K, init_pose, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # result
    min_loss = float("inf")
    best_pose = init_pose.clone()
    for i in range(num_iterations):
        optimizer.zero_grad()
        depth_last = to_tensor(tar_rgb_d.depth, DEVICE)
        depth_current = to_tensor(src_rgb_d.depth, DEVICE)
        loss = model(depth_last, depth_current)
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_pose = model.pose_cur.clone()

        # if i % 10 == 0:
        #     print(f"Iteration {i}: Loss {min_loss}")
        # scheduler.step()
    return min_loss, best_pose


def eval():
    tar_rgb_d, src_rgb_d = Replica()[0], Replica()[1]
    _, estimate_pose = train_model(tar_rgb_d, src_rgb_d, to_tensor(tar_rgb_d.K, DEVICE))

    eT = calculate_translation_error(
        estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose
    )
    eR = calculate_rotation_error(estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose)
    print(f"Translation error: {eT}")
    print(f"Rotation error: {eR}")


if __name__ == "__main__":
    eval()
