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


# 打开输出文件
with open("lilunzhiBianli3.txt", "w") as file:
    # 遍历所有的参数组合
    for alpha in np.arange(0.1, 2.01, 0.01):
        for p00 in np.arange(0.1, 1.0, 0.01): # 并不是从0.01开始的，你想累死你的电脑？
            for p01 in np.arange(0.1, 1.0, 0.01):
                for p10 in np.arange(0.1, 1.0, 0.01):
                    for p11 in np.arange(0.1, 1.0, 0.01):
                        # 求解 beta2 和 theta
                        valid_solutions = solve_beta2_theta(p00, p01, p10, p11, alpha, min_threshold=0.1)

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

                                # 输出结果
                                file.write(f"alpha={alpha}, p00={p00}, p01={p01}, p10={p10}, p11={p11}. result={lambda1}\n")
                        else:
                            # 如果没有有效解，输出默认值99999
                            file.write(f"alpha={alpha}, p00={p00}, p01={p01}, p10={p10}, p11={p11}. result=99999\n")

print("所有计算已完成，结果已保存到 'lilunzhiBianli3.txt' 文件中。")
