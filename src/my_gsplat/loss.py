from typing import Literal

import kornia

from torch import Tensor
from torch.nn import functional as F

from src.my_gsplat.geometry import depth_to_normal


def compute_depth_loss(
    depth_A: Tensor, depth_B: Tensor, *, loss_type: Literal["l1", "mse"] = "l1"
) -> Tensor:
    """
    Compute the mean squared error between two depth images.
    Parameters
    ----------
    depth_A: shape=(*,H,W,*)
    depth_B: shape=(*,H,W,*)
    loss_type: Literal["l1", "mse"]

    Returns
    -------
    loss: Tensor
    """
    if loss_type == "l1":
        return F.l1_loss(depth_A, depth_B)
    elif loss_type == "mse":
        return F.mse_loss(depth_A, depth_B)
    else:
        raise ValueError("Invalid loss type. Use 'mse' or 'l1'.")


def compute_silhouette_loss(
    depth_A: Tensor, depth_B: Tensor, *, loss_type: Literal["l1", "mse"] = "l1"
) -> Tensor:
    """
    Compute the mean squared error between the sobel edges of two depth images.
    Parameters
    ----------
    depth_A: shape=(B, H, W, 1)
    depth_B: shape=(B, H, W, 1)
    loss_type: Literal["l1", "mse"]

    Returns
    -------
    loss: Tensor
    """
    # Ensure the depth images have a batch dimension and a channel dimension
    assert depth_A.dim() == 4 and depth_B.dim() == 4
    # kornia needs shape=(B,C,H,W)
    edge_A = kornia.filters.sobel(depth_A.permute(0, 3, 1, 2))
    edge_B = kornia.filters.sobel(depth_B.permute(0, 3, 1, 2))

    if loss_type == "l1":
        return F.l1_loss(edge_A, edge_B)
    elif loss_type == "mse":
        return F.mse_loss(edge_A, edge_B)
    else:
        raise ValueError("Invalid loss type. Use 'mse', 'l1', or 'huber'.")


def compute_normal_consistency_loss(
    depth_real: Tensor,
    depth_rendered: Tensor,
    *,
    K: Tensor,
    loss_type: Literal["cosine", "l1", "mse"] = "cosine"
) -> Tensor:
    """
    Compute the normal consistency loss between two depth images.

    Args:
        depth_real (Tensor): Real depth image of shape [H, W] or [1, H, W]
        depth_rendered (Tensor): Rendered depth image of shape [H, W] or [1, H, W]
        K: shape=(4,4)
        loss_type (str): Type of loss to use ('cosine', 'l1', or 'mse')

    Returns:
        Tensor: Computed loss
    """
    # Ensure depths are 3D tensors
    if depth_real.dim() == 3:
        depth_real = depth_real.squeeze(0)
    if depth_rendered.dim() == 3:
        depth_rendered = depth_rendered.squeeze(0)

    # Compute normals from depths
    normal_real = depth_to_normal(depth_real, K=K)
    normal_rendered = depth_to_normal(depth_rendered, K=K)

    # Compute loss based on specified type
    if loss_type == "cosine":
        loss = 1 - F.cosine_similarity(normal_real, normal_rendered, dim=1).mean()
    elif loss_type == "l1":
        loss = F.l1_loss(normal_real, normal_rendered)
    elif loss_type == "mse":
        loss = F.mse_loss(normal_real, normal_rendered)
    else:
        raise ValueError("Invalid loss type. Use 'cosine', 'l1', or 'mse'.")

    return loss
