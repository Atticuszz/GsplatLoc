import torch
import torch.nn.functional as F
import torch.optim as optim

from src.component.eval import calculate_translation_error, calculate_rotation_error
from src.slam_data import Replica, RGBDImage
from src.utils import to_tensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class CameraPoseEstimator:
    """
    Initialize the CameraPoseEstimator with camera intrinsics and computation device.

    Parameters
    ----------
    intrinsics : torch.Tensor
        Intrinsic camera matrix with dimensions [3, 3].
    device : str, optional
        The device on which to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
    """

    def __init__(self, intrinsics, device="cuda"):
        self.intrinsics = intrinsics
        self.device = device

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

    def optimize_pose(
        self,
        depth_last,
        pose_last,
        depth_current,
        num_iterations=500,
        learning_rate=1e-3,
    ):
        """
        Optimizes the camera pose by minimizing the difference between the projected and actual depth images.

        Parameters
        ----------
        depth_last : torch.Tensor
            The depth image from the previous frame with dimensions [height, width].
        pose_last : torch.Tensor
            The initial estimate of the pose as a 4x4 transformation matrix.
        depth_current : torch.Tensor
            The depth image from the current frame with dimensions [height, width].
        num_iterations : int, optional
            The number of iterations to run the optimization for. Default is 100.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.01.

        Returns
        -------
        torch.Tensor
            The optimized pose as a 4x4 transformation matrix.
        """
        optimizer = optim.Adam([pose_last], lr=learning_rate)

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            # u,v,pcd
            pcd_last = self.depth_to_pointcloud(depth_last, pose_last)
            pcd_current = self.depth_to_pointcloud(depth_current, pose_last)

            # projected
            projected_depth_last = self.pointcloud_to_depth(pcd_last, pose_last)
            projected_depth_current = self.pointcloud_to_depth(pcd_current, pose_last)

            # combined with closed pcd
            combined_projected_depth = torch.min(
                projected_depth_last, projected_depth_current
            )
            combined_projected_depth[combined_projected_depth == 0] = torch.max(
                projected_depth_last, projected_depth_current
            )[combined_projected_depth == 0]

            # update pose
            loss = F.mse_loss(combined_projected_depth, depth_current)
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss {loss.item()}")

        return pose_last


def eval():
    tar_rgb_d, src_rgb_d = Replica()[0], Replica()[1]
    src_rgb_d: RGBDImage
    tar_rgb_d: RGBDImage
    tracker = CameraPoseEstimator(to_tensor(src_rgb_d.K, device), device)

    estimate_pose = tracker.optimize_pose(
        to_tensor(src_rgb_d.depth, device),
        to_tensor(src_rgb_d.pose, device, requires_grad=True),
        to_tensor(tar_rgb_d.depth, device),
    )

    eT = calculate_translation_error(
        estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose
    )
    eR = calculate_rotation_error(estimate_pose.detach().cpu().numpy(), tar_rgb_d.pose)
    print(f"Translation error: {eT}")
    print(f"Rotation error: {eR}")


if __name__ == "__main__":
    eval()
