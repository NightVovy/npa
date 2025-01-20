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


# 构造矩阵 A0, A1, B0, B1, alphaZ
def construct_matrices_and_alpha(beta2, p00, p01, p10, p11, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    sigma_Z = np.array([[1, 0], [0, -1]])
    sigma_X = np.array([[0, 1], [1, 0]])

    A0 = sigma_Z
    A1 = cos_mu1 * sigma_Z + sin_mu1 * sigma_X
    B0 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X
    B1 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X  # 结构与 B0 一致

    alpha = compute_alpha(p10, p11, sin_mu1, sin_mu2, beta2, theta)
    alphaZ = alpha * sigma_Z

    p00_A0_B0 = p00 * np.dot(A0, B0)
    p01_A0_B1 = p01 * np.dot(A0, B1)
    p10_A1_B0 = p10 * np.dot(A1, B0)
    p11_A1_B1 = p11 * np.dot(A1, B1)

    combined_matrix = alphaZ + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

    return combined_matrix


# 计算特征值和特征向量
def compute_eigenvalues_and_vectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


# 验证一个向量是否是特征向量
def verify_eigenvector(matrix, eigenvalue, vector):
    result = np.dot(matrix, vector)
    expected_result = eigenvalue * vector
    if np.allclose(result, expected_result):
        return True
    else:
        return False


# 示例参数
beta2 = np.pi / 4  # 例如 45 度
p00 = 1.0
p01 = 0.5
p10 = 0.5
p11 = 0.3
theta = np.pi / 6  # 例如 30 度

# 构造矩阵并计算特征值和特征向量
combined_matrix = construct_matrices_and_alpha(beta2, p00, p01, p10, p11, theta)
eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(combined_matrix)

# 输出特征值和特征向量
print("特征值: ", eigenvalues)
print("特征向量: ", eigenvectors)

# 计算测试向量 test_vector = cos(theta)*|00> + sin(theta)*|11>
test_vector = np.array([np.cos(theta), 0, 0, np.sin(theta)])  # 4维列向量

# 验证给定向量是否是特征向量
test_eigenvalue = eigenvalues[0]  # 假设选择第一个特征值
is_eigenvector = verify_eigenvector(combined_matrix, test_eigenvalue, test_vector)
print(f"给定的向量是否是特征向量: {is_eigenvector}")
