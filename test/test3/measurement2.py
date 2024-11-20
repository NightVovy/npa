import numpy as np

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx
I = np.eye(2)  # Identity matrix

# 测量函数
def measure_pure_state(psi, A, B):
    # 计算测量值: <psi|A ⊗ B|psi>
    return np.trace(np.outer(psi, np.conj(psi)) @ np.kron(A, B))
# def measure_pure_state(psi, A0, B0):
#     # 计算A0和B0的张量积
#     operator = np.kron(A0, B0)
#
#     # 计算量子态psi的共轭转置
#     psi_dagger = psi.conj().T
#
#     # 计算测量结果
#     result = psi_dagger @ operator @ psi
#
#     return result

# 主要计算函数
def quantum_measurement(beta2, cos_mu1, cos_mu2, theta, alpha):
    # 使用三角函数计算 sin(mu1) 和 sin(mu2)
    sin_mu1 = np.sqrt(1 - cos_mu1 ** 2)  # 通过 cos(mu1) 计算 sin(mu1)
    sin_mu2 = np.sqrt(1 - cos_mu2 ** 2)  # 通过 cos(mu2) 计算 sin(mu2)

    # 定义量子测量算符
    A0 = sigma_z
    A1 = np.cos(beta2) * sigma_z + np.sin(beta2) * sigma_x
    B0 = cos_mu1 * sigma_z + sin_mu1 * sigma_x
    B1 = cos_mu2 * sigma_z + sin_mu2 * sigma_x

    # 修改量子态 psi 的定义：psi = cos(theta) * [1, 0, 0, 0] + sin(theta) * [0, 0, 0, 1]
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])  # 量子态

    # 计算密度矩阵 TODO:没用
    density_matrix = np.outer(psi, np.conj(psi))

    # 计算测量结果
    A0_measurement = measure_pure_state(psi, A0, I)
    A0B0_measurement = measure_pure_state(psi, A0, B0)
    A0B1_measurement = measure_pure_state(psi, A0, B1)
    A1B0_measurement = measure_pure_state(psi, A1, B0)
    A1B1_measurement = measure_pure_state(psi, A1, B1)

    # # 计算Iap TODO: NOT CORRECT
    # Iap = alpha * A0_measurement + A0B0_measurement + A0B1_measurement + A1B0_measurement - A1B1_measurement
    # # Iap = alpha * A0_measurement + p00 * A0B0_measurement + p01 * A0B1_measurement + p10 * A1B0_measurement - p11 * A1B1_measurement

    # 输出结果
    return A0_measurement, A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement
