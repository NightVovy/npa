import numpy as np
from scipy.optimize import fsolve


def equations(vars, p, q, alpha):
    theta, beta2 = vars

    # 计算 p00, p01, p10, p11
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # 计算 cos(mu1), cos(mu2), sin(mu1), sin(mu2)
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu2 = (-p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    # 方程 1
    eq1 = (p10 * sin_mu1 - p11 * sin_mu2) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta)) - alpha

    # 方程 2
    eq2 = p10 * np.sin(beta2) * cos_mu1 - p11 * np.sin(beta2) * cos_mu2

    return [eq1, eq2]


def find_all_solutions(p, q, alpha, initial_guesses):
    solutions = []

    for guess in initial_guesses:
        try:
            # 求解方程组
            solution = fsolve(equations, guess, args=(p, q, alpha))
            theta, beta2 = solution

            # 确保解在 [0, pi/2] 范围内
            if 0 <= theta <= np.pi / 2 and 0 <= beta2 <= np.pi / 2:
                # 检查解是否已经存在（避免重复解）
                if not any(np.isclose(theta, s[0]) and np.isclose(beta2, s[1]) for s in solutions):
                    solutions.append((theta, beta2))
        except Exception as e:
            # 跳过未能收敛的解
            pass

    return solutions


# 输入参数
p = 0.5
q = 0.35
alpha = 0.6

# 多组初始猜测值
initial_guesses = [
    [0.1, 0.1],
    [0.5, 0.5],
    [np.pi / 4, np.pi / 6],
    [np.pi / 3, np.pi / 3],
    [0.2, np.pi / 8],
    [np.pi / 6, np.pi / 4]
]

# 查找所有可能的解
solutions = find_all_solutions(p, q, alpha, initial_guesses)

# 输出解
if solutions:
    print("找到的所有可能解：")
    for idx, (theta, beta2) in enumerate(solutions):
        print(f"解 {idx + 1}: theta = {theta:.6f}, beta2 = {beta2:.6f}")
else:
    print("未找到任何符合条件的解。")
