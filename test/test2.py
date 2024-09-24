import math

import numpy as np

# 给定的参数
cos_theta = np.cos(np.pi / 4)  # 例如 θ = π/4
sin_theta = np.sin(np.pi / 4)
cos_miu1 = np.cos(np.pi / 6)  # 例如 μ = π/6
sin_miu1 = np.sin(np.pi / 6)
cos_miu2 = np.cos(np.pi / 6)  # 例如 μ = π/6
sin_miu2 = np.sin(np.pi / 6)

# 定义量子态 |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
psi = cos_theta * np.array([1, 0, 0, 0]) + sin_theta * np.array([0, 0, 0, 1])
psi2 = (np.array([0, 1, 0, 0]) - np.array([0, 0, 1, 0])) / math.sqrt(2)

# 定义测量算符 A0, A1, B0, B1
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

A0 = sigma_z
A1 = sigma_x
Q = sigma_z
R = sigma_x
B0 = cos_miu1 * sigma_z + sin_miu1 * sigma_x  # cos(μ1)σz + sin(μ1)σx
B1 = cos_miu2 * sigma_z + sin_miu2 * sigma_x  # cos(μ2)σz + sin(μ2)σx
S = (-sigma_z - sigma_x) / math.sqrt(2)
T = (sigma_z - sigma_x) / math.sqrt(2)


# 计算测量结果
def measure_operator(state, operator):
    return np.vdot(state, operator @ state)
    # 你没发现根本没用上吗 纯态直接用两个向量的共轭点积？

# 计算量子态的密度矩阵
density_matrix2 = np.outer(psi2, psi2.conj())


def measure_pure_state(psi, A0, B0):
    # 计算A0和B0的张量积
    operator = np.kron(A0, B0)

    # 计算量子态psi的共轭转置
    psi_dagger = psi.conj().T

    # 计算测量结果
    result = psi_dagger @ operator @ psi

    return result


# B1_measurement = np.trace(density_matrix @ np.kron(np.eye(2), B1))
# A0S = measure_operator(density_matrix2, np.kron(A0, S))
# A0T = measure_operator(density_matrix2, np.kron(A0, T))
# A1S = measure_operator(density_matrix2, np.kron(A1, S))
QS = measure_pure_state(psi2, Q, S)
RS = measure_pure_state(psi2, R, S)
RT = measure_pure_state(psi2, R, T)
QT = measure_pure_state(psi2, Q, T)

CHSH = QS + RS + RT - QT

print("QS", QS)
print("RS", RS)
print("RT", RT)
print("QT", QT)
print("CHSH = ", CHSH)
