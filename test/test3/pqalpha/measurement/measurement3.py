import numpy as np
from theta_calculator import solve


def quantum_measurements(p, q, alpha):
    # 获取solve函数的返回值
    beta2, theta, sin_beta2, cos_beta2, sin_2theta, cos_2theta, sin_mu1, cos_mu1, sin_mu2, cos_mu2 = solve(p, q, alpha)

    # 定义量子态 psi
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    # 定义泡利矩阵
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])

    # 定义测量算符
    A0 = sigma_z
    A1 = np.cos(beta2) * sigma_z + np.sin(beta2) * sigma_x
    B0 = cos_mu1 * sigma_z + sin_mu1 * sigma_x
    B1 = cos_mu2 * sigma_z + sin_mu2 * sigma_x

    # 定义张量积运算符
    def tensor_product(A, B):
        return np.kron(A, B)

    # 定义计算期望值的函数
    def expectation_value(operator, state):
        return np.vdot(state, operator @ state).real

    # 计算测量值
    results = {}
    I = np.eye(2)
    results['A0I'] = expectation_value(tensor_product(A0, I), psi)
    results['A0B0'] = expectation_value(tensor_product(A0, B0), psi)
    results['A0B1'] = expectation_value(tensor_product(A0, B1), psi)
    results['A1B0'] = expectation_value(tensor_product(A1, B0), psi)
    results['A1B1'] = expectation_value(tensor_product(A1, B1), psi)

    return results


# 示例调用
p = 0.45
q = 0.4
alpha = 0.1
results = quantum_measurements(p, q, alpha)
for key, value in results.items():
    print(f"{key}: {value}")
