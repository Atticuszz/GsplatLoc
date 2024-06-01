from typing import NamedTuple


import torch

from torch import Tensor

from .factor import (
    compute_geometric_residuals,
)
from .pcd import PointClouds
from ..utils import to_tensor


def color_error(source_lab, target_lab):
    """Compute color error in L*a*b* space (can be extended to use CIEDE2000)."""
    return torch.norm(source_lab - target_lab, dim=1).sum()


def jacobian_approx_t(p, f, create_graph=False):
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :param create_graph: If True, the Jacobian will be computed in a differentiable manner.
    :return: jacobian
    """

    try:
        jac = torch.autograd.functional.jacobian(
            f, p, create_graph=create_graph, vectorize=True
        )
    except RuntimeError:
        jac = torch.autograd.functional.jacobian(
            f, p, create_graph=create_graph, strict=True, vectorize=False
        )

    return jac


class TerminationCriteria(NamedTuple):

    ftol: float = 1e-8
    ptol: float = 1e-8
    gtol: float = 1e-8
    tau: float = 1e-3
    meth: str = "lev"
    rho1: float = 0.25
    rho2: float = 0.75
    beta: float = 2
    gama: float = 3
    max_iter: int = 100


class LevenbergMarquardtOptimizer:

    def __init__(
        self,
        init_pose: Tensor,
        tar_pcd: PointClouds,
        src_pcd: PointClouds,
        max_iterations: int = 100,
        init_lambda: float = 1e-3,
        lambda_factor: float = 10,
    ):

        self._init_pose = init_pose
        self._tar_pcd = tar_pcd
        self._src_pcd = src_pcd
        self._max_iterations = max_iterations
        self._lambda = init_lambda  # Damping factor
        self._lambda_factor = lambda_factor  # Damping factor increase/decrease factor

        # result

    def optimize(self):

        for i in range(self._max_iterations):
            f = lambda x: self._error(x, i)
            j = jacobian_approx_t(self._init_pose, f)
            H = torch.matmul(j.T, j)
            b = torch.matmul(j.T, f(self._init_pose))

            # Solve the linearized system of equations
            delta_pose = torch.linalg.solve(
                H + self._lambda * torch.eye(H.shape[0]), -b
            )

            # modify lambda and update pose
            new_pose = self._init_pose + delta_pose
            if self._error(new_pose, i) < self._error(self._init_pose, i):
                self._init_pose = new_pose
                self._lambda /= self._lambda_factor
            else:
                self._lambda *= self._lambda_factor

            # TODO: Check for convergence
            pass

    def _error(
        self,
        T: Tensor,
        src_idx: int,
    ) -> Tensor:
        """
        Compute the GICP error for a single source point against a target point cloud using k-d tree.
        Assumes source and target points include homogeneous coordinates [x, y, z, 1].

        Parameters:
        ----------
        pcd_target : Target point clouds object
        pcd_src : Source point clouds object
        src_idx : Index of the source point
        T : Transformation matrix [4, 4]

        Returns:
        ----------
        error : float, computed Mahalanobis distance or -1 if no neighbor found
        index : int, index of the nearest neighbor or -1 if no neighbor found
        """
        src_point = to_tensor(self._src_pcd.point(src_idx))
        transformed_source = torch.matmul(T, src_point)

        # Assuming the nearest neighbor search is done on the CPU and results are obtained
        found, index, sq_dist = self._tar_pcd.kdtree.nearest_neighbor_search(
            transformed_source.cpu().numpy()
        )
        if not found:
            raise ValueError("No neighbor found!")

        nearest_target_point = to_tensor(self._tar_pcd.point(index))
        residual = nearest_target_point[:3] - transformed_source[:3]
        assert torch.allclose(residual, to_tensor(sq_dist)), "Residuals do not match!"

        error = compute_geometric_residuals(
            nearest_target_point,
            transformed_source,
            T,
            to_tensor(self._tar_pcd.cov(index)),
            to_tensor(self._src_pcd.cov(src_idx)),
        )

        return error
