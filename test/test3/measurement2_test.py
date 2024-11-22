import numpy as np

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

# 测量函数
def measure_pure_state(psi, A, B):
    # 计算测量值: <psi|A ⊗ B|psi>
    return np.trace(np.outer(psi, np.conj(psi)) @ np.kron(A, B))

# 主要计算函数
def quantum_measurement():
    # 定义量子测量算符
    A1 = sigma_x  # Alice的A1操作符，σx

    # Bob的测量操作符
    B0 = (-sigma_z - sigma_x) / np.sqrt(2)  # Bob的B0操作符
    B1 = (sigma_z - sigma_x) / np.sqrt(2)   # Bob的B1操作符

    # 修改量子态 psi 的定义: psi = (1/sqrt(2)) * (|01> - |10>)
    psi = np.array([0, 1, -1, 0]) / np.sqrt(2)  # 量子态

    # 计算测量结果
    A0B0_measurement = measure_pure_state(psi, sigma_z, B0)
    A0B1_measurement = measure_pure_state(psi, sigma_z, B1) # -1/sqrt2
    A1B0_measurement = measure_pure_state(psi, A1, B0)
    A1B1_measurement = measure_pure_state(psi, A1, B1)

    # 输出测量结果
    return A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement

# 计算并显示测量结果
A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement = quantum_measurement()

print(f"A0B0_measurement: {A0B0_measurement}")
print(f"A0B1_measurement: {A0B1_measurement}")
print(f"A1B0_measurement: {A1B0_measurement}")
print(f"A1B1_measurement: {A1B1_measurement}")
