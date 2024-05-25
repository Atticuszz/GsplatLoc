"""
-*- coding: utf-8 -*-
@Organization : SupaVision
@Author       : 18317
@Date Created : 04/02/2024
@Description  :
"""

import numpy as np
from pykdtree.kdtree import KDTree

# 创建一些随机数据点
data_points = np.random.rand(1000, 3)  # 1000个3维点
query_points = np.random.rand(10, 3)  # 10个查询点

# 创建KD树
tree = KDTree(data_points, leafsize=10)

# 执行查询
indices, distances = tree.query(query_points, k=1, sqr_dists=True)

# 打印结果
print("Indices of nearest neighbors:", indices)
print("Squared distances to nearest neighbors:", distances)
