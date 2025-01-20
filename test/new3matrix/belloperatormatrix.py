import numpy as np


# 计算 sin(mu1), sin(mu2), cos(mu1), cos(mu2)
def compute_trig_functions(beta2, p00, p01, p10, p11, theta):
    # 计算 cos(mu1), cos(mu2), sin(mu1), sin(mu2) 使用公式
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


# 构造矩阵 A0, A1, B0, B1, alphaZ
def construct_matrices_and_alpha(beta2, p00, p01, p10, p11, theta):
    # 计算 trig 函数
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    # σ_Z 和 σ_X 矩阵定义
    sigma_Z = np.array([[1, 0], [0, -1]])
    sigma_X = np.array([[0, 1], [1, 0]])

    # 构造矩阵 A0 和 A1
    A0 = sigma_Z
    A1 = cos_mu1 * sigma_Z + sin_mu1 * sigma_X

    # 构造矩阵 B0 和 B1
    B0 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X
    B1 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X  # 注意这与 B0 结构一致

    # 计算 alpha
    alpha = compute_alpha(p10, p11, sin_mu1, sin_mu2, beta2, theta)

    # 构造矩阵 alphaZ
    alphaZ = alpha * sigma_Z

    # 计算矩阵 p00 A0 B0, p01 A0 B1, p10 A1 B0, p11 A1 B1
    p00_A0_B0 = p00 * np.dot(A0, B0)
    p01_A0_B1 = p01 * np.dot(A0, B1)
    p10_A1_B0 = p10 * np.dot(A1, B0)
    p11_A1_B1 = p11 * np.dot(A1, B1)

    # 计算组合矩阵
    combined_matrix = alphaZ + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

    return A0, A1, B0, B1, alphaZ, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1, combined_matrix


# 计算最大特征值
def compute_max_eigenvalue(matrix):
    # 计算特征值
    eigenvalues = np.linalg.eigvals(matrix)
    # 获取最大特征值
    max_eigenvalue = np.max(np.abs(eigenvalues))
    return max_eigenvalue

# 示例参数
beta2 = np.pi / 4  # 例如 45 度
p00 = 1.0
p01 = 0.5
p10 = 0.5
p11 = 0.3
theta = np.pi / 6  # 例如 30 度

# 构造矩阵 A0, A1, B0, B1 和 alphaZ
A0, A1, B0, B1, alphaZ, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1, combined_matrix = construct_matrices_and_alpha(
    beta2, p00, p01, p10, p11, theta)



# 打印结果
print("A0 matrix:\n", A0)
print("A1 matrix:\n", A1)
print("B0 matrix:\n", B0)
print("B1 matrix:\n", B1)
print("alphaZ matrix:\n", alphaZ)

print("\np00 A0 B0 matrix:\n", p00_A0_B0)
print("p01 A0 B1 matrix:\n", p01_A0_B1)
print("p10 A1 B0 matrix:\n", p10_A1_B0)
print("p11 A1 B1 matrix:\n", p11_A1_B1)

# 输出组合矩阵
print("\nCombined matrix (alphaZ + p00A0B0 + p01A0B1 + p10A1B0 - p11A1B1):\n", combined_matrix)

max_eigenvalue = compute_max_eigenvalue(combined_matrix)

print("最大特征值为:", max_eigenvalue)