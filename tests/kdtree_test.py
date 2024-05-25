from timeit import default_timer
import numpy as np
import small_gicp
from scipy.spatial import cKDTree as ScipyKDTree

from src.slam_data import Replica


def generate_random_point_clouds(num_points: int, dimensions: int) -> np.ndarray:
    return np.random.rand(num_points, dimensions).astype(np.float64)


def benchmark_knn_search(
    kdtree_class,
    reference_points: np.ndarray,
    query_points: np.ndarray,
    k: int,
    class_name: str,
):
    start_time = default_timer()

    if class_name == "small_gicp":
        pcd = small_gicp.PointCloud(reference_points)
        kdtree = kdtree_class(pcd, num_threads=32)
    elif class_name == "SciPy":
        kdtree = kdtree_class(
            reference_points, balanced_tree=False, compact_nodes=False
        )
    else:
        kdtree = kdtree_class(reference_points)
    build_time = default_timer() - start_time
    stamps = []
    for q_pcd in query_points:
        start_time = default_timer()
        if class_name == "SciPy":
            indices, distances = kdtree.query(q_pcd, k=k, workers=32)
        elif class_name == "small_gicp":
            indices, distances = kdtree.knn_search(q_pcd, k)
        else:
            indices, distances = kdtree.query(query_points, k=k)
        stamps.append(default_timer() - start_time)
    search_time = np.mean(stamps)
    # print(f"{class_name} build time: {build_time:.16f}")
    # print(f"{class_name} search time: {search_time:.16f}")
    # print("total time:", build_time + search_time)
    return build_time, search_time


k = 1  # For GICP-like behavior
bts, sts = [], []
data_set = Replica()
i = 0
for rgb_d in data_set:
    if i % 100 == 0:
        print(f"{i}/{len(data_set)}")
    i += 1
    pcd = rgb_d.pointclouds(6, include_homogeneous=False)
    bt, st = benchmark_knn_search(small_gicp.KdTree, pcd, pcd, k, "small_gicp")
    bts.append(bt)
    sts.append(st)
print(f"small_gicp mean build time:{np.mean(bts):.16f}")
print(f"small_gicp mean search time:{np.mean(sts):.16f}")
bts, sts = [], []
i = 0
for rgb_d in data_set:
    if i % 100 == 0:
        print(f"{i}/{len(data_set)}")
    i += 1
    pcd = rgb_d.pointclouds(6, include_homogeneous=False)
    bt, st = benchmark_knn_search(ScipyKDTree, pcd, pcd, k, "SciPy")
    bts.append(bt)
    sts.append(st)
print(f"SciPy mean build time:{np.mean(bts):.16f}")
print(f"SciPy mean search time:{np.mean(sts):.16f}")
