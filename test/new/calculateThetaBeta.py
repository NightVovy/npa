import numpy as np
from scipy.optimize import fsolve

def equations(vars, p00, p01, p10, p11, alpha):
    beta2, theta = vars

    # 计算第一个方程的左边
    term1 = p10 * np.sin(beta2) * ((p00 + p10 * np.cos(beta2)) /
                                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (
                                               p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term2 = p11 * np.sin(beta2) * ((p01 - p11 * np.cos(beta2)) /
                                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (
                                               p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation1 = term1 - term2

    # 计算第二个方程的左边
    term3 = p10 * ((p10 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term4 = p11 * ((-p11 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation2 = (term3 - term4) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta)) - alpha

    return [equation1, equation2]

def solve_beta2_theta(p00, p01, p10, p11, alpha, min_threshold=0.1):
    valid_solutions = []

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

    for initial_guess in initial_guesses:
        # 使用 fsolve 来解方程，增加 xtol 容差
        solutions = fsolve(equations, initial_guess, args=(p00, p01, p10, p11, alpha), xtol=1e-6, maxfev=2000)

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

def calculate_mu1_mu2(beta2, theta, p00, p01, p10, p11):
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


def calculate_rhs_I_LHS(beta2, p00, p01, p10, p11, alpha):
    # 计算右侧的公式，即 I_LHS <= sqrt((alpha + p00 + p01 + (p10 - p11) * cos(beta))^2 + (p10 - p11)^2 * sin^2(beta))
    cos_beta = np.cos(beta2)
    sin_beta = np.sin(beta2)

    rhs = np.sqrt(
        (alpha + p00 + p01 + (p10 - p11) * cos_beta) ** 2 +
        (p10 - p11) ** 2 * sin_beta ** 2
    )
    return rhs



# 示例参数
alpha = 0.2
p00 = 0.4
p01 = 0.9
p10 = 0.8
p11 = 0.4


# 输出参数范围
print(f"参数范围：")
print(f"theta 的范围在 (0, {np.pi / 4}) radians")
print(f"beta2, mu1, mu2 的范围在 [0, {np.pi / 2}] radians")

# 求解 beta2 和 theta
valid_solutions = solve_beta2_theta(p00, p01, p10, p11, alpha, min_threshold=0.1)

# 如果找到有效解
if valid_solutions:
    for beta2, theta in valid_solutions:
        # 使用解出的 beta2 和 theta 计算 cos(mu1), sin(mu1), mu1, cos(mu2), sin(mu2), mu2
        cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2 = calculate_mu1_mu2(beta2, theta, p00, p01, p10, p11)

        # 计算 lambda2 值
        lambda2 = np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                  np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                  alpha * np.cos(2 * theta)

        lambda1 = ((p00 + p10 * np.cos(beta2)) * cos_mu1 + (p01 - p11 * np.cos(beta2)) * cos_mu2) + \
                  ((p10 * np.sin(beta2)) * sin_mu1 - (p11 * np.sin(beta2)) * sin_mu2) * np.sin(2 * theta) + \
                  alpha * np.cos(2 * theta)

        # 计算 I_LHS 的右侧值
        rhs_I_LHS = calculate_rhs_I_LHS(beta2, p00, p01, p10, p11, alpha)

        # 输出结果
        print(f"beta2: {beta2} radians, theta: {theta} radians")
        print(f"sin(beta2): {np.sin(beta2)}, sin(2*theta): {np.sin(2 * theta)}")
        print(f"cos(mu1): {cos_mu1}, sin(mu1): {sin_mu1}, mu1: {mu1} radians")
        print(f"cos(mu2): {cos_mu2}, sin(mu2): {sin_mu2}, mu2: {mu2} radians")
        print(f"λ2 (lambda2)A14 bottom = {lambda2}")
        print(f"λ1 (lambda1)A14 top = {lambda1}")
        print(f"Right-hand side of I_LHS <=: {rhs_I_LHS}")
else:
    print("未找到有效的解")

# def generate_second_text(p00_input, p01_input, p10_input, p11_input, alpha):
#     # 根据输入的 p00_input, p01_input, p10_input, p11_input 和 alpha 生成新的文字
#     text = f'npa_max({alpha:.5f} * A1 + {p00_input} * A1 * B1 + {p01_input} * A1 * B2 + {p10_input} * A2 * B1 - {p11_input} * A2 * B2, "1 + A B + A^2 B")'
#
#     return text
#
# result_2 = generate_second_text(p00, p01, p10, p11, alpha)
# print(result_2)