import kornia
import torch
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor

from .utils import DEVICE, knn, rgb_to_sh, to_tensor


def construct_full_pose(rotation: Tensor, translation: Tensor):
    """
    Constructs the full 4x4 transformation matrix from rotation and translation.
    Ensures that gradients are tracked for rotation and translation.
    """
    pose = torch.eye(4, dtype=rotation.dtype, device="cuda:0")
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def compute_silhouette_diff(depth: Tensor, rastered_depth: Tensor) -> Tensor:
    """
    Compute the difference between the sobel edges of two depth images.

    Parameters
    ----------
    depth : torch.Tensor
        The depth image with dimensions [height, width].
    rastered_depth : torch.Tensor
        The depth image with dimensions [height, width].

    Returns
    -------
    torch.Tensor
        The silhouette difference between the two depth images with dimensions [height, width].
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    else:
        depth = depth.unsqueeze(1)
    if rastered_depth.dim() == 2:
        rastered_depth = rastered_depth.unsqueeze(0).unsqueeze(0)
    else:
        rastered_depth = rastered_depth.unsqueeze(1)
    edge_depth = kornia.filters.sobel(depth)
    edge_rastered_depth = kornia.filters.sobel(rastered_depth)
    silhouette_diff = torch.abs(edge_depth - edge_rastered_depth).squeeze()
    return silhouette_diff


def transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Transform points using a SE(3) transformation matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        A 4x4 SE(3) transformation matrix.
    points : torch.Tensor
        An Nx3 tensor of points to be transformed.

    Returns
    -------
    torch.Tensor
        An Nx3 tensor of transformed points.
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return torch.addmm(matrix[:3, 3], points, matrix[:3, :3].t())


def init_gs_scales(
    points: torch.Tensor,
    k: int = 5,
) -> Tensor:
    """
    Initialize scales for Gaussian Splatting with safeguards against large scales.

    Parameters
    ----------
    points: tensor, shape (N, 3)
        Input point cloud
    k: int
        Number of nearest neighbors to consider

    Returns
    -------
    scales: Tensor, shape (N, 3)
        Initialized scales for each point
    """
    dist2_avg = (knn(points, k)[:, 1:] ** 2).mean(dim=-1)
    # dist2_avg = (knn(points, k)[:, 1:] ** 2).median(dim=-1).values
    dist_avg = torch.sqrt(dist2_avg)

    scales = dist_avg.unsqueeze(-1).repeat(1, 3)

    return scales


@torch.no_grad()
def compute_depth_gt(
    points: Tensor,  # N,3
    rgbs: Tensor,  # N,3
    Ks: Tensor,
    c2w: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """

    Parameters
    ----------
        rgbs: N,3
        points: N,3
        Ks: 1,3,3
        c2w: 1,4,4
        height: int
        width: int

    Returns
    -------
        depths: [height, width]
    """
    # Parameters
    means3d = points  # [N, 3]
    init_opa = 1.0
    opacities = torch.logit(
        torch.full((points.shape[0],), init_opa, device=points.device)
    )  # [N,]
    scales = init_gs_scales(points)
    quats = to_tensor([1, 0, 0, 0], device=points.device).repeat(
        points.shape[0], 1
    )  # [N, 4]

    # color is SH coefficients.
    sh_degree = 1
    colors = torch.zeros(
        (points.shape[0], (sh_degree + 1) ** 2, 3), device=DEVICE
    )  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)  # Initialize SH coefficients
    sh0 = colors[:, :1, :]
    shN = colors[:, 1:, :]
    opacities = torch.sigmoid(opacities)
    colors = torch.cat([sh0, shN], 1)
    # colors = torch.sigmoid(self.colors)
    # scales = torch.exp(scales)

    render_colors, _, _ = rasterization(
        means=means3d,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=sh_degree,
        viewmats=torch.linalg.inv(c2w),
        Ks=Ks,
        width=width,
        height=height,
        far_plane=1e10,
        near_plane=1e-2,
        render_mode="ED",
        rasterize_mode="classic",
    )

    depths = render_colors
    return depths.squeeze(0).squeeze(-1)


def depth_to_points(
    depth: Tensor,
    K: Tensor,
    include_homogeneous: bool = False,
) -> Tensor:
    """
    Project depth map to point clouds using intrinsic matrix.

    Parameters
    ----------
    depth: shape=(H,W)
    K: shape=(4,4)
    include_homogeneous: bool, optional
        Whether to include the homogeneous coordinate (default True).

    Returns
    -------
    points: Tensor
        The generated point cloud, shape=(h*w, 3) or (h*w, 4).
    """
    points_3d = kornia.geometry.depth_to_3d_v2(depth, K).view(-1, 3)
    if include_homogeneous:
        points_3d = F.pad(points_3d, (0, 1), value=1)
    return points_3d


def depth_to_normal(depth: Tensor, K: Tensor) -> Tensor:
    """
    Convert depth map to normal map using the provided depth_to_points function.

    Args:
        depth (Tensor): Depth map of shape [H, W]
        K (Tensor): Camera intrinsic matrix of shape [4, 4]

    Returns:
        Tensor: Normal map of shape [H, W, 3]
    """
    H, W = depth.shape

    # Convert depth to 3D points
    points = depth_to_points(depth, K)
    points = points.view(H, W, 3)

    # Add batch dimension
    points = points.unsqueeze(0)  # Now shape is [1, H, W, 3]

    # Compute gradients
    # Use padding to handle border cases
    points_padded = F.pad(points, (0, 0, 1, 1, 1, 1), mode="replicate")

    dx = points_padded[:, 1:-1, 2:, :] - points_padded[:, 1:-1, :-2, :]
    dy = points_padded[:, 2:, 1:-1, :] - points_padded[:, :-2, 1:-1, :]

    # Compute normal vectors using cross product
    normal = torch.cross(dx, dy, dim=-1)

    # Normalize the normal vectors
    normal = F.normalize(normal, p=2, dim=-1)

    return normal.squeeze(0)
