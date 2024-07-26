import torch
from kornia import geometry as KG
from torch import Tensor
from torch.nn import functional as F


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

    normalized_quaternion = KG.normalize_quaternion(quaternion)
    return KG.quaternion_to_rotation_matrix(normalized_quaternion)


def rotation_matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """
    Convert a rotation matrix to a quaternion.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        The rotation matrix with dimensions (..., 3, 3).

    Returns
    -------
    torch.Tensor
        The quaternion with dimensions (*,4).
    """

    return KG.rotation_matrix_to_quaternion(rotation_matrix)
