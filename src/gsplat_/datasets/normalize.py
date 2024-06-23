import torch


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # Rotate the world so that z+ is the up axis
    ups = torch.sum(R * torch.tensor([0, -1.0, 0], device=R.device), dim=-1)
    world_up = torch.mean(ups, dim=0)
    world_up /= torch.norm(world_up)

    up_camspace = torch.tensor([0.0, -1.0, 0.0], device=R.device)
    c = torch.dot(up_camspace, world_up)
    cross = torch.cross(world_up, up_camspace)
    skew = torch.tensor(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        device=R.device,
    )

    if c > -1:
        R_align = torch.eye(3, device=R.device) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=R.device
        )

    R = R_align @ R
    fwds = torch.sum(R * torch.tensor([0, 0.0, 1.0], device=R.device), dim=-1)
    t = (R_align @ t.unsqueeze(-1)).squeeze(-1)

    # Recenter the scene
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(dim=-1).unsqueeze(-1) * fwds
        translate = -torch.median(nearest, dim=0)[0]
    elif center_method == "poses":
        translate = -torch.median(t, dim=0)[0]
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = torch.eye(4, device=R.device)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # Rescale the scene using camera distances
    scale_fn = torch.max if strict_scaling else torch.median
    scale = 1.0 / scale_fn(torch.norm(t + translate, dim=-1))
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    # Compute centroid
    centroid = torch.median(point_cloud, dim=0).values

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = torch.cov(translated_point_cloud.t())

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = eigenvalues.argsort(descending=True)
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant is negative, flip an eigenvector.
    if torch.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.t()

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = torch.eye(4, device=point_cloud.device)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -torch.mv(rotation_matrix, centroid)

    return transform


def transform_points(matrix, points):
    """Transform points using a SE(4) matrix.

    Args:
        matrix: 4x4 SE(4) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return torch.addmm(matrix[:3, 3], points, matrix[:3, :3].t())


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using a SE(4) matrix.

    Args:
        matrix: 4x4 SE(4) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    # Perform the matrix multiplication with einsum for better control
    transformed = torch.einsum("ki,nij->nkj", matrix, camtoworlds)

    # Normalize the 3x3 rotation matrices to maintain scale: Use the norm of the first row
    scaling = torch.norm(transformed[:, 0, :3], p=2, dim=1, keepdim=True)
    transformed[:, :3, :3] /= scaling.unsqueeze(
        -1
    )  # Unsqueeze to match the shape for broadcasting

    return transformed
