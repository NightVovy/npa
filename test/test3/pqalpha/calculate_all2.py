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


def solve_beta2_theta(p, q, alpha, min_threshold=0.1):
    # 设置初值猜测（扩展范围）
    initial_guesses = [
        [np.pi / 4, np.pi / 4],  # 默认初值
        [np.pi / 6, np.pi / 6],  # 不同的初值
        [np.pi / 3, np.pi / 3],  # 不同的初值
        [np.pi / 2, np.pi / 4],  # 更大的初值
        [0.1, 0.2],  # 较小初值
        [np.pi / 4, np.pi / 8],  # 添加新的初值
        [np.pi / 5, np.pi / 3],  # 添加新的初值
        [0.2, np.pi / 6],  # 添加新的初值
        [np.pi / 6, np.pi / 2]  # 添加新的初值
    ]

    valid_solutions = []

    for initial_guess in initial_guesses:
        # 使用 fsolve 来解方程，增加 xtol 容差
        solutions = fsolve(equations, initial_guess, args=(p, q, alpha), xtol=1e-6, maxfev=2000)

        # 如果返回的是一个一维数组，处理它
        if np.ndim(solutions) == 1:
            solutions = [solutions]  # 将其转化为包含一个解的列表

        # 筛选有效解
        for sol in solutions:
            beta2, theta = sol

            # 筛选条件：beta2 和 theta 在合理范围内且大于一个小阈值
            if 0 < beta2 < np.pi / 2 and 0 < theta < np.pi / 4 and theta > min_threshold and beta2 > min_threshold:
                valid_solutions.append((beta2, theta))

    return valid_solutions


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
q = 0.5
alpha = 0.44

# 输出参数范围
print(f"参数范围：")
print(f"theta 的范围在 (0, {np.pi / 4}) radians")
print(f"beta2, mu1, mu2 的范围在 [0, {np.pi / 2}] radians")

# 求解 beta2 和 theta
valid_solutions = solve_beta2_theta(p, q, alpha, min_threshold=0.1)

# 如果找到有效解
if valid_solutions:
    for beta2, theta in valid_solutions:
        # 使用解出的 beta2 和 theta 计算 cos(mu1), sin(mu1), mu1, cos(mu2), sin(mu2), mu2
        cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2 = calculate_mu1_mu2(beta2, theta, p, q)

        # 计算 lambda2 值
        p00 = p * q
        p01 = p * (1 - q)
        p10 = (1 - p) * q
        p11 = (1 - p) * (1 - q)

        lambda2 = np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                  np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                  alpha * np.cos(2 * theta)

        lambda1 = ((p00 + p10 * np.cos(beta2)) * cos_mu1 + (p01 - p11 * np.cos(beta2)) * cos_mu2) + \
                  ((p10 * np.sin(beta2)) * sin_mu1 - (p11 * np.sin(beta2)) * sin_mu2) * np.sin(2 * theta) + \
                  alpha * np.cos(2 * theta)

        # 输出结果
        print(f"beta2: {beta2} radians, theta: {theta} radians")
        print(f"sin(beta2): {np.sin(beta2)}, sin(2*theta): {np.sin(2 * theta)}")
        print(f"cos(mu1): {cos_mu1}, sin(mu1): {sin_mu1}, mu1: {mu1} radians")
        print(f"cos(mu2): {cos_mu2}, sin(mu2): {sin_mu2}, mu2: {mu2} radians")
        print(f"λ2 (lambda2)A14 bottom = {lambda2}")
        print(f"λ1 (lambda1)A14 top = {lambda1}")
else:
    print("未找到有效的解")


# def generate_text(p, q, alpha):
#     # 根据 p 和 q 计算 p00, p01, p10, p11，并保留5位小数
#     p00 = round(p * q, 5)
#     p01 = round(p * (1 - q), 5)
#     p10 = round((1 - p) * q, 5)
#     p11 = round((1 - p) * (1 - q), 5)
#
#     # 根据 alpha、p00、p01、p10、p11 和 layer 生成最终的文字
#     text = f'npa_max({alpha:.5f} * A1 + {p00:.5f} * A1 * B1 + {p01:.5f} * A1 * B2 + {p10:.5f} * A2 * B1 - {p11:.5f} * A2 * B2, "1 + A B + A^2 B")'
#
#     return text
#
# result_1 = generate_text(p, q, alpha)
# print(result_1)