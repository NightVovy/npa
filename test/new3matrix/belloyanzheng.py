import numpy as np


# 计算 sin(mu1), sin(mu2), cos(mu1), cos(mu2)
def compute_trig_functions(beta2, p00, p01, p10, p11, theta):
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu2 = - (p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return cos_mu1, cos_mu2, sin_mu1, sin_mu2


# 计算 alpha
def compute_alpha(p10, p11, sin_mu1, sin_mu2, beta2, theta):
    alpha = (p10 * sin_mu1 - p11 * sin_mu2) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta))
    return alpha


# 构造矩阵 A0, A1, B0, B1, alphaA0
def construct_matrices_and_alpha(beta2, p00, p01, p10, p11, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    # 单量子比特操作矩阵
    sigma_Z = np.array([[1, 0], [0, -1]])
    sigma_X = np.array([[0, 1], [1, 0]])

    # 使用张量积构造矩阵 A0, A1, B0, B1
    A0 = np.cos(beta1) * sigma_Z + np.sin(beta1) * sigma_X # 2x2 矩阵
    A1 = np.cos(beta2) * sigma_Z + np.sin(beta2) * sigma_X  # 2x2 矩阵
    B0 = cos_mu1 * sigma_Z + sin_mu1 * sigma_X  # 2x2 矩阵
    B1 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X  # 2x2 矩阵

    # 计算 alpha
    alpha = compute_alpha(p10, p11, sin_mu1, sin_mu2, beta2, theta)

    # 计算 alphaA0 = alpha * (A0 ⊗ I)
    alphaA0 = alpha * np.kron(A0, np.eye(2))  # alphaA0 = alpha * A0 张量积 2x2 单位矩阵

    # 计算各个矩阵的张量积
    p00_A0_B0 = p00 * np.kron(A0, B0)  # p00 * A0 ⊗ B0
    p01_A0_B1 = p01 * np.kron(A0, B1)  # p01 * A0 ⊗ B1
    p10_A1_B0 = p10 * np.kron(A1, B0)  # p10 * A1 ⊗ B0
    p11_A1_B1 = p11 * np.kron(A1, B1)  # p11 * A1 ⊗ B1

    # 计算组合矩阵
    combination_matrix = alphaA0 + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

    # 计算第一个公式左侧
    left_side_1 = alpha * np.sin(beta1) + (p00 * np.cos(beta1) + p10 * np.cos(beta2)) * sin_mu1 + (p01 * np.cos(beta1) - p11 * np.cos(beta2)) * sin_mu2

    # 计算第二个公式左侧
    left_side_2 = (p00 * np.sin(beta1) + p10 * np.sin(beta2)) * cos_mu1 + (p01 * np.sin(beta1) - p11 * np.sin(beta2)) * cos_mu2

    # 代入简化公式进行计算
    # 第一个简化公式：代入 sin_mu1 和 sin_mu2
    left_side_simplified_1 = (p00 + p10 * np.cos(beta2)) * (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + (p01 - p11 * np.cos(beta2)) * (-p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    # 第二个简化公式：代入 cos_mu1 和 cos_mu2
    left_side_simplified_2 = p10 * np.sin(beta2) * (p00 + p10 * np.cos(beta2)) / np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2) - p11 * np.sin(beta2) * (p01 - p11 * np.cos(beta2)) / np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return left_side_1, left_side_2, left_side_simplified_1, left_side_simplified_2


# 示例参数
beta1 = 0
beta2 = np.pi / 2  # 例如 45 度
p00 = 0.98
p01 = 0.715
p10 = 0.15
p11 = 0.15
theta = np.pi / 6  # 例如 30 度

# 计算左侧结果
left_side_1, left_side_2, left_side_simplified_1, left_side_simplified_2 = construct_matrices_and_alpha(beta2, p00, p01, p10, p11, theta)

# 输出两个公式左侧的计算结果
print("Left side of the first formula:")
print(left_side_1)

print("\nLeft side of the second formula:")
print(left_side_2)

# 输出简化后的两个公式左侧的计算结果
print("\nSimplified Left side of the first formula:")
print(left_side_simplified_1)

print("\nSimplified Left side of the second formula:")
print(left_side_simplified_2)
