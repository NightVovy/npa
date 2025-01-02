import numpy as np
from scipy.optimize import fsolve

def equations(vars, p00, p01, p10, p11, alpha):
    beta2, theta = vars

    term1 = p10 * np.sin(beta2) * ((p00 + p10 * np.cos(beta2)) /
                                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (
                                           p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term2 = p11 * np.sin(beta2) * ((p01 - p11 * np.cos(beta2)) /
                                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (
                                           p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation1 = term1 - term2

    term3 = p10 * ((p10 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    term4 = p11 * ((-p11 * np.sin(beta2) * np.sin(2 * theta)) /
                   np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    equation2 = (term3 - term4) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta)) - alpha

    return [equation1, equation2]

def solve_beta2_theta(p00, p01, p10, p11, alpha, min_threshold=0.1):
    valid_solutions = []
    initial_guesses = [
        [np.pi / 4, np.pi / 4],
        [np.pi / 6, np.pi / 6],
        [np.pi / 3, np.pi / 3],
        [np.pi / 2, np.pi / 4],
        [0.1, 0.2],
        [np.pi / 4, np.pi / 8],
        [np.pi / 5, np.pi / 3],
        [0.2, np.pi / 6],
        [np.pi / 6, np.pi / 2]
    ]

    for initial_guess in initial_guesses:
        solutions = fsolve(equations, initial_guess, args=(p00, p01, p10, p11, alpha), xtol=1e-6, maxfev=2000)
        if np.ndim(solutions) == 1:
            solutions = [solutions]
        for sol in solutions:
            beta2, theta = sol
            if 0 < beta2 < np.pi / 2 and 0 < theta < np.pi / 4 and theta > min_threshold and beta2 > min_threshold:
                valid_solutions.append((beta2, theta))

    return valid_solutions

def calculate_mu1_mu2(beta2, theta, p00, p01, p10, p11):
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    mu1 = np.arccos(cos_mu1)

    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    sin_mu2 = -(p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
    )
    mu2 = np.arccos(cos_mu2)

    return cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2

# 固定 p10 和 p11
p10 = 1.0
p11 = 1.0

# 打开输出文件
with open("lilunBianlip10p111.txt", "w") as file:
    for alpha in np.arange(0.1, 2.1, 0.1):
        for p00 in np.arange(0.1, 0.91, 0.1):
            for p01 in np.arange(0.1, 0.91, 0.1):
                valid_solutions = solve_beta2_theta(p00, p01, p10, p11, alpha, min_threshold=0.1)

                if valid_solutions:
                    for beta2, theta in valid_solutions:
                        cos_mu1, sin_mu1, mu1, cos_mu2, sin_mu2, mu2 = calculate_mu1_mu2(beta2, theta, p00, p01, p10, p11)
                        lambda2 = np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                                  np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2) + \
                                  alpha * np.cos(2 * theta)
                        lambda1 = ((p00 + p10 * np.cos(beta2)) * cos_mu1 + (p01 - p11 * np.cos(beta2)) * cos_mu2) + \
                                  ((p10 * np.sin(beta2)) * sin_mu1 - (p11 * np.sin(beta2)) * sin_mu2) * np.sin(2 * theta) + \
                                  alpha * np.cos(2 * theta)
                        file.write(f"alpha={alpha}, p00={p00}, p01={p01}, p10={p10}, p11={p11}. result = {lambda1}\n")
                else:
                    file.write(f"alpha={alpha}, p00={p00}, p01={p01}, p10={p10}, p11={p11}. result = 99999\n")

print("所有计算已完成，结果已保存到 'lilunBianlip10p111.txt' 文件中。")
