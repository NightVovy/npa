import numpy as np
import cvxpy as cp
from scipy.optimize import root_scalar

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

# 量子测量函数
def measure_pure_state(psi, A, B):
    # 计算测量值: <psi|A ⊗ B|psi>
    return np.trace(np.outer(psi, np.conj(psi)) @ np.kron(A, B))

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

    # 计算密度矩阵
    density_matrix = np.outer(psi, np.conj(psi))

    # 计算测量结果
    A0_measurement = np.trace(density_matrix @ np.kron(A0, np.eye(2)))
    A0B0_measurement = measure_pure_state(psi, A0, B0)
    A0B1_measurement = measure_pure_state(psi, A0, B1)
    A1B0_measurement = measure_pure_state(psi, A1, B0)
    A1B1_measurement = measure_pure_state(psi, A1, B1)

    # 计算Iap
    Iap = alpha * A0_measurement + A0B0_measurement + A0B1_measurement + A1B0_measurement - A1B1_measurement

    # 输出结果，每个值后面换行
    return (f"A0 = {A0_measurement}\n"
            f"A0B0 = {A0B0_measurement}\n"
            f"A0B1 = {A0B1_measurement}\n"
            f"A1B0 = {A1B0_measurement}\n"
            f"A1B1 = {A1B1_measurement}\n"
            f"Iap = {Iap}")

# Pauli and SDP functions
def calculate_beta2(p, q, theta, epsilon=1e-10):
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    def beta2_equation(beta2):
        sin_beta2 = np.sin(beta2)
        cos_beta2 = np.cos(beta2)

        numerator1 = p00 + p10 * cos_beta2
        denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator1 < epsilon:
            return None
        cos_mu1 = numerator1 / denominator1

        numerator2 = p01 - p11 * cos_beta2
        denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator2 < epsilon:
            return None
        cos_mu2 = numerator2 / denominator2

        return p10 * sin_beta2 * cos_mu1 - p11 * sin_beta2 * cos_mu2

    try:
        result = root_scalar(beta2_equation, bracket=[0.1, np.pi - 0.1], method='brentq')
        if result.converged and 0 < result.root < np.pi - epsilon:
            beta2 = result.root
            cos_beta2 = np.cos(beta2)
            sin_beta2 = np.sin(beta2)

            numerator1 = p00 + p10 * cos_beta2
            denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu1 = numerator1 / max(denominator1, epsilon)

            numerator2 = p01 - p11 * cos_beta2
            denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu2 = numerator2 / max(denominator2, epsilon)

            sin_mu1 = np.sqrt(1 - cos_mu1**2)
            sin_mu2 = np.sqrt(1 - cos_mu2**2)

            alpha = (p10 * sin_mu1 - p11 * sin_mu2) * sin_beta2 * (np.cos(2 * theta) / np.sin(2 * theta))

            if 0 < alpha < 2:
                mu1 = np.arccos(cos_mu1)
                mu2 = np.arccos(cos_mu2)

                is_mu1_greater_than_pi_over_4 = mu1 > np.pi / 4
                is_mu2_greater_than_pi_over_4 = mu2 > np.pi / 4

                return beta2, cos_mu1, cos_mu2, alpha, mu1, mu2, is_mu1_greater_than_pi_over_4, is_mu2_greater_than_pi_over_4
    except:
        pass

    return None

# SDP solver function
def calculate_sdp(p, q, alpha):
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    gamma = cp.Variable((9, 9))
    constraints = [
        gamma >> 0,
        gamma == cp.conj(gamma.T),
        cp.diag(gamma) == 1,

        gamma[0, 1] == gamma[3, 5],
        gamma[0, 1] == gamma[4, 6],
        gamma[0, 2] == gamma[3, 7],
        gamma[0, 2] == gamma[4, 8],
        gamma[0, 3] == gamma[1, 5],
        gamma[0, 3] == gamma[2, 7],
        gamma[0, 4] == gamma[2, 8],
        gamma[0, 4] == gamma[1, 6],
        gamma[1, 3] == gamma[0, 5],
        gamma[1, 4] == gamma[0, 6],
        gamma[2, 3] == gamma[0, 7],
        gamma[2, 4] == gamma[0, 8],
        gamma[1, 2] == gamma[5, 7],
        gamma[1, 2] == gamma[6, 8],
        gamma[3, 4] == gamma[5, 6],
        gamma[3, 4] == gamma[7, 8],
        gamma[2, 5] == gamma[1, 7],
        gamma[2, 6] == gamma[1, 8],
        gamma[4, 5] == gamma[3, 6],
        gamma[3, 8] == gamma[4, 7],
        gamma[5, 7] == gamma[6, 8],
    ]

    alpha_param = cp.Parameter(value=alpha)

    objective = cp.Maximize(
        alpha_param * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4]
    )

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SDPA")

    return gamma.value, problem.value

# 遍历 p, q, theta
p_values = np.arange(0.05, 1.05, 0.05)
q_values = np.arange(0.05, 1.05, 0.05)
theta_values = np.arange(0.1, np.pi / 4, 0.05)

found_results = False  # 标记是否找到符合条件的结果

for p in p_values:
    for q in q_values:
        for theta in theta_values:
            result = calculate_beta2(p, q, theta)
            if result:
                beta2, cos_mu1, cos_mu2, alpha, mu1, mu2, is_mu1_greater_than_pi_over_4, is_mu2_greater_than_pi_over_4 = result
                gamma_matrix, lambda_value = calculate_sdp(p, q, alpha)

                A0B0 = gamma_matrix[1, 3]
                A0B1 = gamma_matrix[1, 4]
                A1B0 = gamma_matrix[2, 3]
                A1B1 = gamma_matrix[2, 4]

                A0B0_measurement = measure_pure_state(np.array([np.cos(theta), 0, 0, np.sin(theta)]), sigma_z, sigma_z)
                A0B1_measurement = measure_pure_state(np.array([np.cos(theta), 0, 0, np.sin(theta)]), sigma_z, sigma_x)
                A1B0_measurement = measure_pure_state(np.array([np.cos(theta), 0, 0, np.sin(theta)]), sigma_x, sigma_z)
                A1B1_measurement = measure_pure_state(np.array([np.cos(theta), 0, 0, np.sin(theta)]), sigma_x, sigma_x)

                if np.isclose(A0B0, A0B0_measurement, atol=1e-3) and np.isclose(A0B1, A0B1_measurement, atol=1e-3) and \
                        np.isclose(A1B0, A1B0_measurement, atol=1e-3) and np.isclose(A1B1, A1B1_measurement, atol=1e-3):
                    found_results = True
                    print(f"Found matching results: p={p}, q={q}, theta={theta}, beta2={beta2}, alpha={alpha}, "
                          f"cos_mu1={cos_mu1}, cos_mu2={cos_mu2}, A0B0={A0B0}, A0B1={A0B1}, A1B0={A1B0}, A1B1={A1B1}")
                    print(f"Gamma matrix:\n{gamma_matrix}")
                    print(f"Optimal lambda: {lambda_value}")
                    break
        if found_results:
            break
    if found_results:
        break

if not found_results:
    print("No results found.")
