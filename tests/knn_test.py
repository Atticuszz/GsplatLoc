from timeit import default_timer

import numpy as np
import torch
from scipy.spatial import KDTree
from src.gicp.pcd import PointClouds
from src.pose_estimation.gemoetry import KnnSearch
from src.utils import to_tensor


def compute_accuracy(truth, predictions):
    """计算两个数组匹配的百分比。"""
    # 使用 np.isclose 对数组中的每个元素进行比较，这会返回一个布尔数组
    matches = np.isclose(truth, predictions)
    # 计算匹配的百分比
    accuracy = np.mean(matches) * 100  # 转换为百分比
    return accuracy


class Knntest:
    def __init__(self, n_points=1000, dim=3):
        self.pcd = np.random.rand(n_points, dim)

    def test_knn(self, k=5):
        # small_gicp kdtree
        custom_pcd = PointClouds(self.pcd)
        custom_pcd.preprocess(knn=k)
        start_time = default_timer()
        custom_knn_time = default_timer() - start_time
        custom_results = custom_pcd.kdtree.batch_knn_search(self.pcd, k)

        # SciPy kdtree
        scipy_kdtree = KDTree(self.pcd)
        start_time = default_timer()
        scipy_results = scipy_kdtree.query(self.pcd, k=k)
        scipy_knn_time = default_timer() - start_time

        # pytorch
        torch_knn = KnnSearch(to_tensor(self.pcd, device="cuda", dtype=torch.float32))
        start_time = default_timer()
        torch_results = torch_knn.search(
            to_tensor(self.pcd, device="cuda", dtype=torch.float32), k
        )
        torch_knn_time = default_timer() - start_time

        # compare results
        custom_indices = np.array(custom_results[0])
        scipy_indices = scipy_results[1]
        torch_indices = torch_results.cpu().numpy()
        small_gicp_accuracy = compute_accuracy(custom_indices, scipy_indices)
        torch_knn_accuracy = compute_accuracy(custom_indices, torch_indices)
        return {
            "Custom KdTree Time": custom_knn_time,
            "SciPy KdTree Time": scipy_knn_time,
            "torch Time": torch_knn_time,
            "Small GICP Accuracy": small_gicp_accuracy,
            "torch_knn Accuracy": torch_knn_accuracy,
        }


if __name__ == "__main__":
    # 测试
    test_pcd = Knntest(n_points=810000, dim=3)
    results = test_pcd.test_knn(k=5)
    print(results)
    # results = test_pcd.test_nns()
    # print(results)
