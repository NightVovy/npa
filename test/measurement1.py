import numpy as np

# 给定的参数
cos_theta = np.cos(np.pi/4)  # 例如 θ = π/4
sin_theta = np.sin(np.pi/4)
cos_miu = np.cos(np.pi/6)    # 例如 μ = π/6
sin_miu = np.sin(np.pi/6)

# 定义量子态 |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
psi = cos_theta * np.array([1, 0, 0, 0]) + sin_theta * np.array([0, 0, 0, 1])

# 定义测量算符 A0, A1, B0, B1
A0 = np.array([[1, 0], [0, -1]])  # σz
A1 = np.array([[0, 1], [1, 0]])   # σx
B0 = cos_miu * np.array([[1, 0], [0, -1]]) + sin_miu * np.array([[0, 1], [1, 0]])  # cos(μ)σz + sin(μ)σx
B1 = -cos_miu * np.array([[1, 0], [0, -1]]) - sin_miu * np.array([[0, 1], [1, 0]]) # -cos(μ)σz - sin(μ)σx

# 计算测量结果
def measure_operator(state, operator):
    return np.vdot(state, operator @ state)

A0_measurement = measure_operator(psi, A0)
A1_measurement = measure_operator(psi, A1)
B0_measurement = measure_operator(psi, B0)
B1_measurement = measure_operator(psi, B1)

print("A0 测量结果:", A0_measurement)
print("A1 测量结果:", A1_measurement)
print("B0 测量结果:", B0_measurement)
print("B1 测量结果:", B1_measurement)
