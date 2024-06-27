import kornia.geometry as KG
import torch
import torch.nn.functional as F
from torch import Tensor


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


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
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


def generate_depth_map(
    points: Tensor, c2w: Tensor, K: Tensor, image_size: tuple
) -> Tensor:
    """
    Generate a depth map from normalized point cloud.

    Parameters:
    - points (torch.Tensor): The normalized 3D point cloud, shape (N, 3).
    - c2w (torch.Tensor): The camera-to-world transformation matrix, shape (4, 4).
    - K (torch.Tensor): The intrinsic camera matrix, shape (3, 3).
    - image_size (tuple): The dimensions of the target image (height, width).

    Returns:
    - torch.Tensor: The generated depth map, shape (image_size[0], image_size[1]).
    """
    # Invert c2w to get world-to-camera transformation
    w2c = torch.linalg.inv(c2w)

    # Transform points from world to camera coordinates
    points_hom = torch.cat(
        (
            points,
            torch.ones(points.size(0), 1, dtype=points.dtype, device=points.device),
        ),
        dim=1,
    )
    points_cam = torch.mm(points_hom, w2c.t())[:, :3]

    # Project points onto the image plane
    projected = torch.mm(points_cam, K.t())
    projected[:, :2] /= projected[:, 2].unsqueeze(1)  # Normalize x, y by z (depth)

    # Initialize depth map with zeros
    depth_map = torch.zeros(image_size, dtype=torch.float32, device=points.device)

    # Convert projected points into discrete image coordinates
    x = projected[:, 0].long()
    y = projected[:, 1].long()

    # Filter points within the image bounds
    mask = (x >= 0) & (x < image_size[1]) & (y >= 0) & (y < image_size[0])
    x = x[mask]
    y = y[mask]
    depth_values = points_cam[:, 2][mask]

    # Fill depth map
    depth_map[y, x] = depth_values

    return depth_map