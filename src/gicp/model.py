from timeit import default_timer

import torch
import torch.nn as nn
import torch.optim as optim

from src.gicp.pcd import PointClouds
from src.slam_data import Replica, RGBDImage
from src.utils import to_tensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# class GICPModel(nn.Module):
#     def __init__(
#         self,
#         tar_pcd: PointClouds,
#         src_pcd: PointClouds,
#         init_T: torch.Tensor | None = None,
#     ):
#         super().__init__()
#         if init_T is not None:
#             self.transformation = nn.Parameter(init_T)
#         else:
#             self.transformation = nn.Parameter(
#                 torch.eye(4, dtype=torch.float64, device=device)
#             )
#         self.tar_pcd = tar_pcd
#         self.src_pcd = src_pcd
#         with torch.no_grad():
#             self.src = to_tensor(src_pcd.points)
#             self.covs_target = to_tensor(tar_pcd.covs)
#             self.covs_src = to_tensor(src_pcd.covs)
#
#     def forward(
#         self,
#         target: torch.Tensor,
#     ) -> torch.Tensor:
#         errors = []
#         for i in range(len(self.src_pcd)):
#             try:
#                 error = self._error(target, i)
#                 errors.append(error)
#             except ValueError as e:
#                 # Handle the case where no neighbor is found
#                 print(e)
#                 continue
#         total_loss = torch.stack(errors).mean() if errors else torch.tensor(0.0)
#         return total_loss
#
#     def _error(
#         self,
#         target: torch.Tensor,
#         src_idx: int,
#     ) -> Tensor:
#         """
#         Compute the GICP error for a single source point against a target point cloud using k-d tree.
#         Assumes source and target points include homogeneous coordinates [x, y, z, 1].
#
#         Parameters:
#         ----------
#         pcd_target : Target point clouds object
#         pcd_src : Source point clouds object
#         src_idx : Index of the source point
#         T : Transformation matrix [4, 4]
#
#         Returns:
#         ----------
#         error : float, computed Mahalanobis distance or -1 if no neighbor found
#         index : int, index of the nearest neighbor or -1 if no neighbor found
#         """
#         src_point = self.src[src_idx]
#         transformed_source = torch.matmul(self.transformation, src_point)
#
#         # Assuming the nearest neighbor search is done on the CPU and results are obtained
#         found, index, sq_dist = self.tar_pcd.kdtree.nearest_neighbor_search(
#             transformed_source.detach().cpu().numpy()[:3]
#         )
#         if not found:
#             raise ValueError("No neighbor found!")
#
#         nearest_target_point = target[index]
#         residual = nearest_target_point[:3] - transformed_source[:3]
#         # assert torch.allclose(residual, to_tensor(sq_dist)), "Residuals do not match!"
#
#         error = self.compute_mahalanobis_loss(
#             residual,
#             self.covs_target[index],
#             self.covs_src[src_idx],
#         )
#
#         return error
#
#     def compute_mahalanobis_loss(
#         self,
#         residual: Tensor,
#         cov_target: Tensor,
#         cov_src: Tensor,
#     ) -> torch.Tensor:
#         combined_covariance = cov_target + torch.matmul(
#             torch.matmul(self.transformation, cov_src),
#             self.transformation.transpose(0, 1),
#         )
#         inv_combined_covariance = torch.inverse(combined_covariance[:3, :3])
#
#         # Calculate Mahalanobis distance
#         error = 0.5 * torch.matmul(
#             torch.matmul(residual.unsqueeze(0), inv_combined_covariance),
#             residual.unsqueeze(1),
#         )
#         return error


class GICPModel(nn.Module):
    def __init__(
        self,
        tar_pcd: PointClouds,
        src_pcd: PointClouds,
        init_T: torch.Tensor | None = None,
    ):
        super().__init__()
        self.transformation = nn.Parameter(
            init_T
            if init_T is not None
            else torch.eye(4, dtype=torch.float64, device=device)
        )
        self.src_pcd = src_pcd
        self.tar_pcd = tar_pcd
        self.src_points = to_tensor(src_pcd.points, device=device)
        self.tar_points = to_tensor(tar_pcd.points, device=device)
        self.covs_src = to_tensor(src_pcd.covs, device=device)
        self.covs_tar = to_tensor(tar_pcd.covs, device=device)
        self.batch_size = (
            500  # Adjust the batch size based on your GPU/CPU memory availability
        )

    def forward(self) -> torch.Tensor:
        num_batches = (self.src_points.size(0) + self.batch_size - 1) // self.batch_size
        errors = []

        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, self.src_points.size(0))
            batch_src_points = self.src_points[batch_start:batch_end]

            transformed_batch_src = torch.matmul(
                batch_src_points, self.transformation.T
            )
            dists = torch.cdist(transformed_batch_src, self.tar_points)
            min_dists, indices = torch.min(dists, dim=1)

            batch_errors = self.compute_mahalanobis_loss(
                min_dists, indices, batch_start
            )
            errors.append(batch_errors)

        total_loss = (
            torch.cat(errors).mean() if errors else torch.tensor(0.0, device=device)
        )
        return total_loss

    def compute_mahalanobis_loss(self, min_dists, indices, batch_start):
        # 提取对应的协方差矩阵
        covs_src = self.covs_src[batch_start : batch_start + indices.size(0)]
        covs_tar = self.covs_tar[indices]

        # 计算组合协方差矩阵和其逆
        T = self.transformation
        transformed_covs_src = torch.matmul(
            torch.matmul(T, covs_src), T.transpose(0, 1)
        )
        combined_covariances = covs_tar + transformed_covs_src
        inv_combined_covariances = torch.linalg.inv(
            combined_covariances[:, :3, :3]
        )  # 只对空间部分取逆

        # 计算残差
        src_points_transformed = torch.matmul(
            self.src_points[batch_start : batch_start + indices.size(0)], T
        )
        residuals = self.tar_points[indices, :3] - src_points_transformed[:, :3]

        # 计算每个点的马氏距离
        mahalanobis_distances = torch.sum(
            (residuals.unsqueeze(1) @ inv_combined_covariances) * residuals, dim=2
        )

        # 返回马氏距离的平均作为损失
        loss = torch.mean(mahalanobis_distances)
        return loss


def training(
    tar_pcd: PointClouds,
    src_pcd: PointClouds,
    num_epochs: int = 100,
) -> float:
    model = GICPModel(tar_pcd, src_pcd)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        start = default_timer()
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        end = default_timer() - start

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Time: {end:.6f}s")
    return loss.item()


def eval():
    tar_rgb_d, src_rgb_d = Replica()[0], Replica()[1]
    src_rgb_d: RGBDImage
    tar_rgb_d: RGBDImage
    src_pcd = PointClouds(src_rgb_d.color_pcds())
    tar_pcd = PointClouds(tar_rgb_d.color_pcds())

    src_pcd.preprocess(20)
    tar_pcd.preprocess(20)

    loss = training(
        src_pcd,
        tar_pcd,
        num_epochs=5,
    )
    print(f"Final loss: {loss}")


if __name__ == "__main__":
    eval()
