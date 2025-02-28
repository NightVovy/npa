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

    # 计算 sin(beta2), sin(2*theta), cos(2*theta)
    sin_beta2 = np.sin(beta2)
    cos_beta2 = np.cos(beta2)
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)

    return beta2, theta, sin_beta2, cos_beta2, sin_2theta, cos_2theta


def calculate_mu1_mu2(beta2, theta, p, q):
    # 根据给定公式计算 p00, p01, p10, p11
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # 计算 cos(mu1) 和 sin(mu1)
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    mu1 = np.arccos(cos_mu1)  # 通过 cos(mu1) 计算 mu1

    # 计算 cos(mu2) 和 sin(mu2)
    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    sin_mu2 = -(p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    mu2 = np.arccos(cos_mu2)  # 通过 cos(mu2) 计算 mu2，确保 mu2 > 0

    return cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2


# 示例参数
p = 0.5
q = 0.4
alpha = 0.8

# 求解 beta2 和 theta
beta2, theta, sin_beta2, cos_beta2, sin_2theta, cos_2theta = solve_beta2_theta(p, q, alpha)

# 使用解出的 beta2 和 theta 计算 cos(mu1), sin(mu1), mu1, cos(mu2), sin(mu2), mu2
cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2 = calculate_mu1_mu2(beta2, theta, p, q)


# 计算 lambda 值
p00 = p * q
p01 = p * (1 - q)
p10 = (1 - p) * q
p11 = (1 - p) * (1 - q)


# 计算 lambda1 值
lambda1 = (
    (p00 + p10 * cos_beta2) * cos_mu1 + (p01 - p11 * cos_beta2) * cos_mu2
) + (
    (p10 * sin_beta2) * sin_mu1 - (p11 * sin_beta2) * sin_mu2
) * sin_2theta + alpha * cos_2theta

lambda2 = np.sqrt((p00 + p10 * cos_beta2)**2 + (p10 * sin_beta2 * sin_2theta)**2) + \
                np.sqrt((p01 - p11 * cos_beta2)**2 + (p11 * sin_beta2 * sin_2theta)**2) + \
                alpha * cos_2theta


# 输出参数范围
print(f"参数范围：")
print(f"theta 的范围在 (0, {np.pi/4}) radians")
print(f"beta2, mu1, mu2 的范围在 [0, {np.pi/2}] radians")


# 输出结果
print(f"beta2: {beta2} radians, theta: {theta} radians")
print(f"sin(beta2): {sin_beta2}, cos(beta2): {cos_beta2}, sin(2*theta): {sin_2theta}, cos(2*theta): {cos_2theta}")
print(f"cos(mu1): {cos_mu1}, sin(mu1): {sin_mu1}, mu1: {mu1} radians")
print(f"cos(mu2): {cos_mu2}, sin(mu2): {sin_mu2}, mu2: {mu2} radians")
print(f"\nλ1 (lambda1) = {lambda1}")
print(f"λ2 (lambda2) = {lambda2}")
