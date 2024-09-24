import numpy as np

from functions import measure_pure_state

# 给定的参数
cos_theta = np.cos(np.pi / 4)  # 例如 θ = π/4
sin_theta = np.sin(np.pi / 4)
cos_miu1 = np.cos(np.pi / 6)  # 例如 μ = π/6
sin_miu1 = np.sin(np.pi / 6)
cos_miu2 = np.cos(np.pi / 6)  # 例如 μ = π/6
sin_miu2 = np.sin(np.pi / 6)
alpha = 0

# 定义量子态 |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
psi = cos_theta * np.array([1, 0, 0, 0]) + sin_theta * np.array([0, 0, 0, 1])

# 定义测量算符 A0, A1, B0, B1
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

A0 = sigma_z
A1 = sigma_x
B0 = cos_miu1 * sigma_z + sin_miu1 * sigma_x  # cos(μ1)σz + sin(μ1)σx
B1 = cos_miu2 * sigma_z + sin_miu2 * sigma_x  # cos(μ2)σz + sin(μ2)σx


# 计算量子态的密度矩阵
density_matrix = np.outer(psi, psi.conj())

# 计算测量结果
A0_measurement = np.trace(density_matrix @ np.kron(A0, np.eye(2)))
# A0B0_measurement = np.trace(density_matrix @ np.kron(A0, B0))
# A0B1_measurement = np.trace(density_matrix @ np.kron(A0, B1))
# A1B0_measurement = np.trace(density_matrix @ np.kron(A1, B0))
# A1B1_measurement = np.trace(density_matrix @ np.kron(A1, B1))
# B1_measurement = np.trace(density_matrix @ np.kron(np.eye(2), B1))
A0B0_measurement = measure_pure_state(psi, A0, B0)
A0B1_measurement = measure_pure_state(psi, A0, B1)
A1B0_measurement = measure_pure_state(psi, A1, B0)
A1B1_measurement = measure_pure_state(psi, A1, B1)

Iap = alpha * A0_measurement + A0B0_measurement + A0B1_measurement + A1B0_measurement - A1B1_measurement

print("A0 测量结果:", A0_measurement)
print("A0B0 测量结果:", A0B0_measurement)
print("A0B1 测量结果:", A0B1_measurement)
print("A1B0 测量结果:", A1B0_measurement)
print("A1B1 测量结果:", A1B1_measurement)
print("Iap = ", Iap)
