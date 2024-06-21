import logging
import warnings
from timeit import default_timer

import numpy as np
import torch
from scipy.spatial import KDTree


from src.gsplat_.utils import KnnSearch

# from src.pose_estimation.geometry import KnnSearchKeOp3
from src.utils import to_tensor


def compute_accuracy(truth, predictions):
    """计算两个数组匹配的百分比。"""
    # 使用 np.isclose 对数组中的每个元素进行比较，这会返回一个布尔数组
    matches = np.isclose(truth, predictions)
    # 计算匹配的百分比
    accuracy = np.mean(matches) * 100  # 转换为百分比
    return accuracy


def test_knn(k=5):
    pcd = np.random.rand(810000, 3)
    # small_gicp kdtree
    pcd_tensor = to_tensor(pcd, device="cpu", dtype=torch.float64)
    knn_tensor = KnnSearch(pcd_tensor)
    start_time = default_timer()
    tensor_knn_results = knn_tensor.query(pcd_tensor, k)
    tensor_knn_time = default_timer() - start_time

    # numpy_knn
    numpy_knn = KnnSearch(pcd)
    start_time = default_timer()
    numpy_knn_res = numpy_knn.query(pcd, k=k)
    numpy_knn_time = default_timer() - start_time

    # SciPy kdtree
    scipy_kdtree = KDTree(pcd)
    start_time = default_timer()
    scipy_results = scipy_kdtree.query(pcd, k=k)
    scipy_knn_time = default_timer() - start_time

    # # keOp3
    # KeOp3 = KnnSearchKeOp3(pcd_tensor)
    # start_time = default_timer()
    # # KeOp3_results = KeOp3.search(pcd_tensor, k)
    # KeOp3_time = default_timer() - start_time

    # compare results
    tensor_indices = tensor_knn_results[0].cpu().numpy()
    numpy_indices = numpy_knn_res[0]
    scipy_indices = scipy_results[1]
    # KeOp3_results_indices = KeOp3_results.cpu().numpy()

    tensor_indices_accuracy = compute_accuracy(tensor_indices, scipy_indices)
    numpy_indices_accuracy = compute_accuracy(numpy_indices, scipy_indices)
    # KeOp3_accuracy = compute_accuracy(scipy_indices, KeOp3_results_indices)
    assert tensor_indices_accuracy == 100
    assert numpy_indices_accuracy == 100
    # assert KeOp3_accuracy == 100
    res = {
        "tensor KdTree Time": tensor_knn_time,
        "numpy KdTree Time": numpy_knn_time,
        "SciPy KdTree Time": scipy_knn_time,
        # "KeOp3 Time": KeOp3_time,
        "tensor Accuracy": tensor_indices_accuracy,
        "numpy Accuracy": numpy_indices_accuracy,
        # "KeOp3 Accuracy": KeOp3_accuracy,
    }
    print(res)
