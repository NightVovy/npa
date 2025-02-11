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
    A0 = np.cos(beta1) * sigma_Z + np.sin(beta1) * sigma_X  # 2x2 矩阵
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

    return alpha, alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1


# 示例参数
beta1 = 0
p00 = 0.66682436795075
p01 = 0.786291363523314
p10 = 0.409866957749884
p11 = 0.50059661260185
beta2 = 0.912451472991185
theta = 0.749780770512441

# 构造矩阵
alpha, alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1 = construct_matrices_and_alpha(beta2, p00, p01, p10, p11,
                                                                                          theta)
# 计算 costheta00 + sintheta11
# 量子态右矢 |00> 和 |11> 分别对应向量 [1, 0, 0, 0] 和 [0, 0, 0, 1]
state_00 = np.array([1, 0, 0, 0])  # |00> 对应的向量
state_11 = np.array([0, 0, 0, 1])  # |11> 对应的向量

origin_state = np.cos(theta) * state_00 + np.sin(theta) * state_11  # 原始态

# 输出 theta 和 beta2 的角度
theta_deg = np.degrees(theta)
beta2_deg = np.degrees(beta2)

# 验证角度是否在 (0, pi/4) 和 (0, pi/2) 之间
theta_check = 0 < theta < np.pi / 2
beta2_check = 0 < beta2 < np.pi / 2




# 输出 alpha 和各个矩阵
print("alpha: ", alpha)
print("\nalphaA0 (张量I后):")
print(alphaA0)
print("\np00A0B0:")
print(p00_A0_B0)
print("\np01A0B1:")
print(p01_A0_B1)
print("\np10A1B0:")
print(p10_A1_B0)
print("\np11A1B1:")
print(p11_A1_B1)

# 组合矩阵
combination_matrix = alphaA0 + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

# 输出组合矩阵
print("\n组合矩阵 alphaA0 + p00A0B0 + p01A0B1 + p10A1B0 - p11A1B1:")
print(combination_matrix)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(combination_matrix)

# 找到最大特征值的索引
max_eigenvalue_index = np.argmax(eigenvalues)

# 提取最大特征值对应的特征向量
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

# 输出最大特征值和对应的特征向量
print("\n最大特征值:", eigenvalues[max_eigenvalue_index])
print("\n对应的特征向量:")
print(max_eigenvector)
print("\ncostheta00 + sintheta11:", origin_state)
print(f"theta (角度): {theta_deg:.2f}°")
print(f"beta2 (角度): {beta2_deg:.2f}°")
print(f"theta 是否在 (0, pi/4) 区间内: {0 < theta < np.pi / 4}")
print(f"beta2 是否在 (0, pi/2) 区间内: {0 < beta2 < np.pi / 2}")