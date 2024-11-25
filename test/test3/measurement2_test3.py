import numpy as np

# 定义Pauli矩阵
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# 输入参数
theta = np.pi / 8
beta2 = np.pi / 8
mu1 = np.pi / 8
mu2 = np.pi / 8

# 量子态 |psi⟩
psi = np.cos(theta) * np.array([1, 0, 0, 0]) + np.sin(theta) * np.array([0, 0, 0, 1])

# 测量算符
A0 = sigma_z
A1 = np.cos(beta2) * sigma_z + np.sin(beta2) * sigma_x
B0 = np.cos(mu1) * sigma_z + np.sin(mu1) * sigma_x
B1 = np.cos(mu2) * sigma_z - np.sin(mu2) * sigma_x

# 计算期望值⟨psi|A0B0|psi⟩等
def compute_expectation(psi, A, B):
    A_kron_B = np.kron(A, B)
    psi_ket = psi.reshape(4, 1)
    psi_bra = psi_ket.conj().T
    result = psi_bra @ A_kron_B @ psi_ket
    return result[0, 0]

E_A0B0 = compute_expectation(psi, A0, B0)
E_A0B1 = compute_expectation(psi, A0, B1)
E_A1B0 = compute_expectation(psi, A1, B0)
E_A1B1 = compute_expectation(psi, A1, B1)

# 计算 Bell 不等式的值
CHSH_value = E_A0B0 + E_A0B1 + E_A1B0 - E_A1B1

print(f'⟨psi|A0B0|psi⟩ = {E_A0B0}')
print(f'⟨psi|A0B1|psi⟩ = {E_A0B1}')
print(f'⟨psi|A1B0|psi⟩ = {E_A1B0}')
print(f'⟨psi|A1B1|psi⟩ = {E_A1B1}')
print(f'CHSH 值 = {CHSH_value}')
