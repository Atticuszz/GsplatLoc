import numpy as np
import torch

from src.gsplat_.datasets import normalize, normalize_np


def test_normalize_functions():
    # Create random camera-to-world matrices
    num_cameras = 10
    c2w_np = np.random.randn(num_cameras, 4, 4)
    c2w_np[:, :3, 3] *= 10  # scale translations

    # Convert to PyTorch tensor
    c2w_torch = torch.tensor(c2w_np, dtype=torch.float32)

    # Normalize using NumPy
    transform_np = normalize_np.similarity_from_cameras(c2w_np)

    # Normalize using PyTorch
    transform_torch = normalize.similarity_from_cameras(c2w_torch)

    # Compare results
    transform_np_torch = torch.tensor(transform_np, dtype=torch.float32)
    if torch.allclose(transform_torch, transform_np_torch, atol=1e-6):
        print("The normalization results are consistent between NumPy and PyTorch.")
    else:
        print("There is a discrepancy between the NumPy and PyTorch results.")


def test_transform_functions():
    num_points = 810000
    num_cameras = 5

    # Random points and camera matrices
    points_np = np.random.randn(num_points, 3)
    camtoworlds_np = np.random.randn(num_cameras, 4, 4)
    matrix_np = np.random.randn(4, 4)

    # Convert to PyTorch tensors
    points_torch = torch.tensor(points_np, dtype=torch.float32)
    camtoworlds_torch = torch.tensor(camtoworlds_np, dtype=torch.float32)
    matrix_torch = torch.tensor(matrix_np, dtype=torch.float32)

    # Run NumPy functions
    transformed_points_np = normalize_np.transform_points(matrix_np, points_np)
    transformed_cameras_np = normalize_np.transform_cameras(matrix_np, camtoworlds_np)

    # Run PyTorch functions
    transformed_points_torch = normalize.transform_points(matrix_torch, points_torch)
    transformed_cameras_torch = normalize.transform_cameras(
        matrix_torch, camtoworlds_torch
    )

    # Compare results
    assert torch.allclose(
        torch.tensor(transformed_points_np, dtype=torch.float32),
        transformed_points_torch,
        atol=1e-6,
    )
    assert torch.allclose(
        torch.tensor(transformed_cameras_np, dtype=torch.float32),
        transformed_cameras_torch,
        atol=1e-6,
    )
    print("Transformation functions are consistent between NumPy and PyTorch.")


def test_align_principle_axes():
    num_points = 810000
    points_np = np.random.rand(num_points, 3) * 100

    # Convert to PyTorch tensor
    points_torch = torch.tensor(points_np, dtype=torch.float32)

    # Run NumPy function
    transform_np = normalize_np.align_principle_axes(points_np)

    # Run PyTorch function
    transform_torch = normalize.align_principle_axes(points_torch)

    # Convert NumPy result to tensor for comparison
    transform_np_torch = torch.tensor(transform_np, dtype=torch.float32)

    # Compare results
    if torch.allclose(transform_torch, transform_np_torch, atol=1e-6):
        print("Alignment functions are consistent between NumPy and PyTorch.")
    else:
        print("Discrepancy found between NumPy and PyTorch alignment functions.")


if __name__ == "__main__":
    for _ in range(1000):
        test_normalize_functions()
        test_transform_functions()
        test_align_principle_axes()
