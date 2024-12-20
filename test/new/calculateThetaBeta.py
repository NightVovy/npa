import numpy as np
from scipy.optimize import fsolve


# 计算p00, p01, p10, p11
def calculate_p_values(p, q):
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)
    return p00, p01, p10, p11


# 定义方程组
def equations(vars, p, q, alpha):
    theta, beta2 = vars
    p00, p01, p10, p11 = calculate_p_values(p, q)

    # 第一个方程
    eq1 = p10 * ((p10 * np.sin(beta2) * np.sin(2 * theta)) /
                 np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)) - \
          p11 * (- (p11 * np.sin(beta2) * np.sin(2 * theta)) /
                 np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)) - alpha

    # 第二个方程
    eq2 = p10 * np.sin(beta2) * ((p00 + p10 * np.cos(beta2)) /
                                 np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (
                                             p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)) - \
          p11 * np.sin(beta2) * ((p01 - p11 * np.cos(beta2)) /
                                 np.sqrt(
                                     (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

    return [eq1, eq2]


# 主函数
def solve_system(p, q, alpha):
    # 设定初始猜测值 (theta, beta2)
    initial_guess = [np.pi / 8, np.pi / 4]  # 在合适范围内选取初值

    # 调用fsolve求解方程组
    solution = fsolve(equations, initial_guess, args=(p, q, alpha))
    theta, beta2 = solution

    # 返回结果，确保theta和beta2在指定的范围内
    if 0 < theta < np.pi / 4 and 0 <= beta2 <= np.pi / 2:
        return theta, beta2
    else:
        return None, None


# 测试
p = 0.7
q = 0.5
alpha = 0.6

theta, beta2 = solve_system(p, q, alpha)
if theta is not None and beta2 is not None:
    print(f"theta: {theta:.4f}, beta2: {beta2:.4f}")
else:
    print("没有找到满足条件的解。")
