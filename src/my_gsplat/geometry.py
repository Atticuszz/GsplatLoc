import kornia
import kornia.geometry as KG
import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import DEVICE


@torch.compile
def construct_full_pose(rotation: Tensor, translation: Tensor):
    """
    Constructs the full 4x4 transformation matrix from rotation and translation.
    Ensures that gradients are tracked for rotation and translation.
    """
    pose = torch.eye(4, dtype=rotation.dtype, device="cuda:0")
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


@torch.compile
def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """
    Convert a rotation matrix to a quaternion.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        The rotation matrix with dimensions [3, 3].

    Returns
    -------
    torch.Tensor
        The quaternion with dimensions [4].
    """
    return KG.rotation_matrix_to_quaternion(rotation_matrix)


@torch.compile
def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.compile
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique. Adapted from pytorch3d.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@torch.compile
def quat_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """
    Convert a quaternion to a rotation matrix.

    Parameters
    ----------
    quaternion : torch.Tensor
        The quaternion with dimensions [4].

    Returns
    -------
    torch.Tensor
        The rotation matrix with dimensions [3, 3].
    """

    normalized_quaternion = quaternion / torch.norm(quaternion)
    return KG.quaternion_to_rotation_matrix(normalized_quaternion)


@torch.compile
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


def add_background_and_penalize_depth(
    image: Tensor,
    depth: Tensor,
    alphas: Tensor,
    background_color: Tensor = torch.tensor([1.0, 1.0, 1.0], device=DEVICE),
):
    """
    对透明区域添加背景色，并对深度为0的区域施加高惩罚。

    参数:
    - image: 彩色图像，形状为 [B, H, W, C]。
    - depth: 深度图，形状为 [B, H, W, 1]。
    - alphas: 透明度图，形状为 [B, H, W, 1]。
    - background_color: 背景颜色，形状为 [C]。
    - depth_penalty: 深度为0的区域的高惩罚值。

    返回:
    - updated_image: 更新后的彩色图像。
    - updated_depth: 更新后的深度图。
    """
    depth_penalty = torch.max(depth).item()
    background_color = background_color.view(1, 1, 1, -1).expand_as(image)
    transparent_mask = alphas == 0
    zero_depth_mask = depth == 0
    updated_image = torch.where(
        transparent_mask.expand_as(image), background_color, image
    )
    updated_depth = torch.where(
        zero_depth_mask, torch.full_like(depth, depth_penalty), depth
    )
    return updated_image, updated_depth


@torch.compile
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
