from timeit import default_timer

import numpy as np
from scipy.spatial import KDTree
from src.gicp.pcd import PointClouds


class Knntest:
    def __init__(self, n_points=1000, dim=3):
        self.pcd = np.random.rand(n_points, dim)

    def test_knn(self, k=5):
        # 创建和预处理自定义 KdTree
        custom_pcd = PointClouds(self.pcd)

        custom_pcd.preprocess(knn=k)
        start_time = default_timer()
        custom_knn_time = default_timer() - start_time
        custom_results = custom_pcd.kdtree.batch_knn_search(self.pcd, k)

        # 创建和使用 SciPy 的 KdTree
        scipy_kdtree = KDTree(self.pcd)
        start_time = default_timer()
        scipy_results = scipy_kdtree.query(self.pcd, k=k)
        scipy_knn_time = default_timer() - start_time

        # 比较结果
        custom_indices = np.array(custom_results[0])
        scipy_indices = scipy_results[1]
        accuracy = np.allclose(custom_indices, scipy_indices)

        return {
            "Custom KdTree Time": custom_knn_time,
            "SciPy KdTree Time": scipy_knn_time,
            "Accuracy": accuracy,
        }

    def test_nns(self):
        # 创建和预处理自定义 KdTree
        custom_pcd = PointClouds(self.pcd)

        custom_pcd.preprocess(knn=1)
        start_time = default_timer()
        custom_knn_time = default_timer() - start_time
        custom_results = custom_pcd.kdtree.batch_nns_search(self.pcd)

        # 创建和使用 SciPy 的 KdTree
        scipy_kdtree = KDTree(self.pcd)
        start_time = default_timer()
        scipy_results = scipy_kdtree.query(self.pcd, k=1)
        scipy_knn_time = default_timer() - start_time

        # 比较结果
        custom_indices = np.array(custom_results[0])
        scipy_indices = scipy_results[1]
        accuracy = np.allclose(custom_indices, scipy_indices)

        return {
            "Custom KdTree Time": custom_knn_time,
            "SciPy KdTree Time": scipy_knn_time,
            "Accuracy": accuracy,
        }


if __name__ == "__main__":
    # 测试
    test_pcd = Knntest(n_points=810000, dim=3)
    results = test_pcd.test_knn(k=5)
    print(results)
    results = test_pcd.test_nns()
    print(results)
