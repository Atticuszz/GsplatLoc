import kornia
from kornia.losses import InverseDepthSmoothnessLoss
from torch import Tensor
from torch.nn import functional as F


def compute_depth_loss(
    depth_A: Tensor, depth_B: Tensor, loss_type: str = "l1"
) -> Tensor:
    """Compute the mean squared error between two depth images."""
    if loss_type == "mse":
        return F.mse_loss(depth_A, depth_B)
    elif loss_type == "l1":
        return F.l1_loss(depth_A, depth_B)
    elif loss_type == "InverseDepthSmoothnessLoss":
        if depth_A.dim() == 2:
            depth_A = depth_A.unsqueeze(0).unsqueeze(
                0
            )  # Reshape [H, W] to [1, 1, H, W]
        if depth_B.dim() == 2:
            depth_B = depth_B.unsqueeze(0).unsqueeze(
                0
            )  # Reshape [H, W] to [1, 1, H, W]
        smooth = InverseDepthSmoothnessLoss()
        return smooth(depth_A, depth_B)
    else:
        raise ValueError("Invalid loss type. Use 'mse' or 'l1'.")


def compute_silhouette_loss(
    depth_A: Tensor, depth_B: Tensor, loss_type: str = "l1"
) -> Tensor:
    """Compute the mean squared error between the sobel edges of two depth images."""
    # Ensure the depth images have a batch dimension and a channel dimension
    if depth_A.dim() == 2:
        depth_A = depth_A.unsqueeze(0).unsqueeze(0)  # Reshape [H, W] to [1, 1, H, W]
    if depth_B.dim() == 2:
        depth_B = depth_B.unsqueeze(0).unsqueeze(0)  # Reshape [H, W] to [1, 1, H, W]

    edge_A = kornia.filters.sobel(depth_A)
    edge_B = kornia.filters.sobel(depth_B)
    if loss_type == "mse":
        return F.mse_loss(edge_A, edge_B)
    elif loss_type == "l1":
        return F.l1_loss(edge_A, edge_B)
    else:
        raise ValueError("Invalid loss type. Use 'mse' or 'l1'.")
