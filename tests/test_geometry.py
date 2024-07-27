import torch

from src.my_gsplat.transform import matrix_to_rotation_6d, rotation_6d_to_matrix


def test_rotation_conversion():
    # 创建随机的旋转矩阵
    num_matrices = 10000  # 测试的矩阵数量
    random_rotations = torch.randn(num_matrices, 3, 3)

    # 正规化这些矩阵使其成为合法的旋转矩阵
    for i in range(num_matrices):
        q, _ = torch.linalg.qr(random_rotations[i])
        random_rotations[i] = q

    # 将旋转矩阵转换为6D表示
    rotation_6d = matrix_to_rotation_6d(random_rotations)

    # 将6D表示转换回旋转矩阵
    reconstructed_matrices = rotation_6d_to_matrix(rotation_6d)

    # 计算原始矩阵和重建矩阵之间的误差
    error = torch.norm(random_rotations - reconstructed_matrices, dim=(1, 2))

    # 检查误差是否在一个合理的范围内（这里使用1e-5作为容错率）
    assert torch.all(error < 1e-5), f"Test failed with error: {error}"

    print("Test passed!")


if __name__ == "__main__":
    # 运行测试函数
    test_rotation_conversion()
