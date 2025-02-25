import numpy as np
from scipy.optimize import minimize


def optimize_pijsame():
    solutions = []
    while len(solutions) < 5:  # 确保有5组有效数据
        initial_guess = np.array([
            np.random.uniform(0.1, 0.9),  # p 取 (0.1, 0.9)
            np.random.uniform(-0.9, 0.9),  # cosbeta2 取 (-1, 1)
            np.random.uniform(0.1, 0.9)  # cos2theta 取 (0.1, 0.9)
        ])

        result = minimize(left_side, initial_guess, bounds=[
            (0.1, 0.9),  # p 的范围
            (-0.9, 0.9),  # cosbeta2 的范围
            (0.1, 0.9)  # cos2theta 的范围
        ])

        if result.success:
            p_opt, cosbeta2_opt, cos2theta_opt = result.x

            # 确保 p, cosbeta2, cos2theta 不接近边界
            if 0.1 < p_opt < 0.9 and -0.9 < cosbeta2_opt < 0.9 and 0.1 < cos2theta_opt < 0.9:
                solutions.append((p_opt, cosbeta2_opt, cos2theta_opt))

    return np.array(solutions)  # 返回 (p, cosbeta2, cos2theta)


def left_side(params):
    p, cosbeta2, cos2theta = params
    term1 = p * (p + p * cosbeta2) / np.sqrt(
        (p + p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    term2 = p * (p - p * cosbeta2) / np.sqrt(
        (p - p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    return abs(term1 - term2)
