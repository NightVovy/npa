import numpy as np
from scipy.optimize import fsolve


def equations(vars, p, q, alpha):
    beta2, theta = vars

    # 根据给定的公式计算 p00, p01, p10, p11
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # 计算第一个方程的左边
    term1 = p10 * np.sin(beta2) * ((p00 + p10 * np.cos(beta2)) /
                                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (
                                               p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term2 = p11 * np.sin(beta2) * ((p01 - p11 * np.cos(beta2)) /
                                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (
                                               p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation1 = term1 - term2

    # 根据给定的 alpha 计算第二个方程的左边
    term3 = p10 * ((p10 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term4 = p11 * ((-p11 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation2 = (term3 - term4) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta)) - alpha

    return [equation1, equation2]


def solve_beta2_theta(p, q, alpha):
    # 设置初值猜测
    initial_guess = [np.pi / 4, np.pi / 4]  # 初始猜测值

    # 使用fsolve来解方程
    beta2, theta = fsolve(equations, initial_guess, args=(p, q, alpha))

    # 计算 sin(beta2) 和 sin(2*theta)
    sin_beta2 = np.sin(beta2)
    sin_2theta = np.sin(2 * theta)

    return beta2, theta, sin_beta2, sin_2theta


# 示例参数
p = 0.55
q = 0.4
alpha = 0.1

# 求解 beta2 和 theta 以及它们的 sin 值
beta2, theta, sin_beta2, sin_2theta = solve_beta2_theta(p, q, alpha)

# 输出结果
print(f"beta2: {beta2} radians, theta: {theta} radians")
print(f"sin(beta2): {sin_beta2}, sin(2*theta): {sin_2theta}")
