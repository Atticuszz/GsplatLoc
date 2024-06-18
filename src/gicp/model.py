from timeit import default_timer

import torch
import torch.nn as nn
import torch.optim as optim
from kornia.geometry.conversions import vector_to_skew_symmetric_matrix as skew
from torch.autograd import Function

from src.gicp.pcd import PointClouds
from src.pose_estimation.geometry import (
    construct_full_pose,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from src.slam_data import Replica, RGBDImage
from src.utils import to_tensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class GICPJacobianApprox(Function):
    @staticmethod
    def forward(ctx, transformation, src_points, tar_points, covs_src, covs_tar):
        # RCR
        transformed_covs_src = torch.matmul(
            torch.matmul(transformation, covs_src),
            transformation.transpose(0, 1),
        )
        RCR = covs_tar + transformed_covs_src
        inv_RCR = torch.linalg.inv(RCR[:, :3, :3])

        residuals = tar_points[:, :3] - src_points[:, :3]

        # 计算每个点的马氏距离
        # residuals 形状是 (N, 3)
        # inv_RCR 形状是 (N, 3, 3)
        # 想要得到的结果是每个点的马氏距离，形状应为 (N,)
        residuals = residuals.unsqueeze(-1)  # 改变形状为 (N, 3, 1) 以适配矩阵乘法
        mahalanobis_distances = torch.matmul(
            residuals.transpose(-2, -1), torch.matmul(inv_RCR, residuals)
        ).squeeze()

        total_loss = 0.5 * mahalanobis_distances.mean()

        ctx.save_for_backward(
            transformation,
            src_points,
            tar_points,
            covs_src,
            covs_tar,
            residuals,
            inv_RCR,
        )
        return total_loss

    # TODO: bug here
    @staticmethod
    def backward(ctx, grad_output):
        (
            transformation,
            src_points,
            tar_points,
            covs_src,
            covs_tar,
            residuals,
            inv_RCR,
        ) = ctx.saved_tensors

        # Compute skew-symmetric matrices for each source point
        skew_matrix = skew(src_points[:, :3])

        # j_r.shape = (n,3,3)
        J_rotation = -torch.einsum("ik,nkj->nij", transformation[:3, :3], skew_matrix)
        # print(f"J_rotation:{J_rotation.shape}")
        # j_t.shape = (n,3,3)
        J_translation = -torch.eye(3, device=src_points.device).expand(
            src_points.size(0), 3, 3
        )

        J = torch.cat([J_rotation, J_translation], dim=-1)  # shape (N, 3, 6)

        # Efficient calculation of grad_residuals
        # grad shape n,
        # residuals shape n,3,1
        grad_residuals = torch.einsum(
            "nji,njk,nkl->n", residuals.transpose(-2, -1), inv_RCR, residuals
        ).squeeze()

        # Sum up gradients for rotation and translation separately and then average them
        # Compute the gradient of transformation
        grad_transformation = torch.zeros_like(transformation)
        grad_transformation[:3, :3] = (
            torch.einsum("nij,n->ij", J[:, :, :3], grad_residuals)
            * grad_output
            / src_points.size(0)
        )
        grad_transformation[:3, 3] = (
            torch.einsum("nij,n->j", J[:, :, :3], grad_residuals)
            * grad_output
            / src_points.size(0)
        )

        return grad_transformation, None, None, None, None


class GICPModel(nn.Module):
    def __init__(
        self,
        tar_pcd: PointClouds,
        src_pcd: PointClouds,
        init_T: torch.Tensor | None = None,
    ):
        super().__init__()
        # optimization Parameter
        if init_T is None:
            init_T = torch.eye(4, dtype=torch.float64, device=device)
        self.quaternion = nn.Parameter(rotation_matrix_to_quaternion(init_T[:3, :3]))
        self.translation = nn.Parameter(init_T[:3, 3])

        # self.transformation = nn.Parameter(
        #     init_T
        #     if init_T is not None
        #     else torch.eye(4, dtype=torch.float64, device=device)
        # )

        # tensorize the point clouds
        self.src_points = to_tensor(src_pcd.points, device=device)
        self.tar_points = to_tensor(tar_pcd.points, device=device)
        # self.tar_knn = KnnSearch(self.tar_points[:, :3].contiguous())
        self.tar_knn = tar_pcd.kdtree
        self.covs_src = to_tensor(src_pcd.covs, device=device)
        self.covs_tar = to_tensor(tar_pcd.covs, device=device)

    def forward(self) -> torch.Tensor:
        rotation = quaternion_to_rotation_matrix(self.quaternion)
        transformation = construct_full_pose(rotation, self.translation)
        # batch_knn for preprocessed point clouds
        transformed_src_points = torch.matmul(self.src_points, transformation)
        nearest_indices, _ = self.tar_knn.batch_nearest_neighbor_search(
            transformed_src_points[:, :3].detach().cpu().numpy()
        )
        nearest_indices = to_tensor(nearest_indices, device=device, dtype=torch.long)

        # select the nearest points and covariances
        selected_tar_points = self.tar_points[nearest_indices]
        covs_src = self.covs_src
        covs_tar = self.covs_tar[nearest_indices]

        return GICPJacobianApprox.apply(
            transformation,
            transformed_src_points,
            selected_tar_points,
            covs_src,
            covs_tar,
        )


# def training(
#     tar_pcd: PointClouds,
#     src_pcd: PointClouds,
#     num_epochs: int = 100,
# ) -> float:
#     model = GICPModel(tar_pcd, src_pcd)
#
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(
#         optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=5, mode="triangular"
#     )
#
#     lr = scheduler.get_last_lr()
#
#     for epoch in range(num_epochs):
#         print(f"last lr: {lr}")
#         start = default_timer()
#         optimizer.zero_grad()
#         loss = model()
#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 print(f"Gradient of {name} has norm: {param.grad.norm().item()}")
#
#         optimizer.step()
#         scheduler.step(loss)
#         end = default_timer() - start
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Time: {end:.6f}s")
#     return loss.item()


def training(tar_pcd, src_pcd, num_epochs=1000):
    model = GICPModel(tar_pcd, src_pcd)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1e-2,
        max_iter=50,
        history_size=200,
        line_search_fn="strong_wolfe",
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name} has norm: {param.grad.norm().item()}")
        lr = optimizer.param_groups[0]["lr"]
        print(f"last lr: {lr}")
        start = default_timer()
        optimizer.step(closure)  # 注意：LBFGS 需要一个闭包来重新计算模型
        loss = closure()  # 重新计算损失
        # scheduler.step(loss)
        end = default_timer() - start

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Time: {end:.6f}s\n"
        )

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
        num_epochs=100,
    )
    print(f"Final loss: {loss}")


if __name__ == "__main__":
    eval()
