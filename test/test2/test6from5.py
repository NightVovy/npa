import numpy as np
import cvxpy as cp
from scipy.optimize import fsolve
import math

# 定义变量
gamma = cp.Variable((9, 9))

# 定义约束条件
constraints = [
    gamma >> 0,  # 半正定约束
    gamma == cp.conj(gamma.T),  # 共轭转置对称
    cp.diag(gamma) == 1,  # 对角线元素为1

    gamma[0, 1] == gamma[3, 5],  # A0
    gamma[0, 1] == gamma[4, 6],

    gamma[0, 2] == gamma[3, 7],  # A1
    gamma[0, 2] == gamma[4, 8],

    gamma[0, 3] == gamma[1, 5],  # 第一行第四个元素等于第2行第六个元素 B0
    gamma[0, 3] == gamma[2, 7],

    gamma[0, 4] == gamma[2, 8],  # B1
    gamma[0, 4] == gamma[1, 6],

    gamma[1, 3] == gamma[0, 5],  # A0B0

    gamma[1, 4] == gamma[0, 6],  # A0B1

    gamma[2, 3] == gamma[0, 7],  # A1B0

    gamma[2, 4] == gamma[0, 8],  # A1B1

    gamma[1, 2] == gamma[5, 7],  # A0A1
    gamma[1, 2] == gamma[6, 8],
    gamma[3, 4] == gamma[5, 6],  # B0B1
    gamma[3, 4] == gamma[7, 8],

    gamma[2, 5] == gamma[1, 7],  # A0A1B0
    gamma[2, 6] == gamma[1, 8],  # A0A1B1
    gamma[4, 5] == gamma[3, 6],  # A0B0B1
    gamma[3, 8] == gamma[4, 7],  # A1B0B1

    gamma[5, 7] == gamma[6, 8],  # X4

]

# 定义参数
p00 = 0.8
p01 = 0.1
p10 = 0.9
p11 = 0.6
alpha = 0.8

# 目标函数
objective = cp.Maximize(
    alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

# 定义并求解问题
problem = cp.Problem(objective, constraints)

# 求解问题
problem.solve(solver="SDPA")  # SDPA OR mosek

# 先输出矩阵
print("Optimal gamma:", gamma.value)

# Define equations
def equations(vars, p00, p01, p10, p11, alpha):
    beta2, theta = vars
    A = p00 + p10 * np.cos(beta2)
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * np.cos(beta2)
    D = p11 * np.sin(beta2) * np.sin(2 * theta)

    if abs(np.cos(beta2)) >= 1 or abs(np.cos(2 * theta)) >= 1:
        return [np.inf, np.inf]  # Return infinity if conditions are not met

    eq1 = (p10 * np.sin(beta2) * A / np.sqrt(A ** 2 + B ** 2)) - (p11 * np.sin(beta2) * C / np.sqrt(C ** 2 + D ** 2))
    eq2 = ((p10 ** 2 * np.sin(beta2) / np.sqrt(A ** 2 + B ** 2)) + (p11 ** 2 * np.sin(beta2) / np.sqrt(C ** 2 + D ** 2))) * np.cos(2 * theta) - alpha

    return [eq1, eq2]

def update(p00, p01, p10, p11, alpha):
    initial_guess = [0.5, 0.5]
    beta2, theta = fsolve(equations, initial_guess, args=(p00, p01, p10, p11, alpha), maxfev=10000)
    cos_beta2 = np.cos(beta2)
    A = p00 + p10 * cos_beta2
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * cos_beta2
    D = p11 * np.sin(beta2) * np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    lambda_val = np.sqrt(A ** 2 + B ** 2) + np.sqrt(C ** 2 + D ** 2) + alpha * cos_2theta
    # print(f"beta2: {beta2}, theta: {theta}, cos(beta2): {cos_beta2}, A: {A}, B: {B}, C: {C}, D: {D}, cos(2*theta): {cos_2theta}")
    print(f"p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}, alpha: {alpha}")
    print(f"A: {A}, B: {B}, C: {C}, D: {D}")
    print(f"beta2: {beta2}, theta: {theta}")
    print(f"cos(2*theta): {cos_2theta}, cos(beta2): {cos_beta2}")
    print(f"lambda: {lambda_val}")

# Call the update function
update(p00, p01, p10, p11, alpha)

# 输出结果
print("NPA Optimal value:", problem.value)

