import numpy as np
import cvxpy as cp
import math
from itertools import product


def find_best_parameters():
    # 定义固定参数
    p10 = 1
    p11 = 1

    # 定义 p00 和 p01 的取值范围
    p_values = np.linspace(0, 1, 100)  # 100个点从0到1
    alpha_values = np.linspace(0, 2, 100)  # 100个点从0到2

    # 遍历所有可能的 p00, p01, alpha 组合
    for p00, p01, alpha in product(p_values, repeat=3):
        if p00 + p01 > 1:
            continue  # 确保 p00 + p01 <= 1

        # 计算 cosbeta2 和 alpha2
        cosbeta2 = (p01 - p00) / (p11 + p10)

        if abs(cosbeta2) >= 1:
            continue  # 确保 |cosbeta2| < 1

        alpha2 = ((p11 ** 2) * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 +
                  2 * p01 * p11 * (p11 + p10) * (p01 - p00)) / (
                         (p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2)

        if alpha ** 2 >= alpha2:
            continue  # 确保 alpha^2 < alpha2

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
            gamma[0, 3] == gamma[1, 5],  # B0
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

        # 计算 F 和 lambda1
        F = (((p00 + p10 * cosbeta2) * p10) / ((p01 - p11 * cosbeta2) * p11)) + 1
        lambda1 = F * math.sqrt(
            (1 + alpha ** 2 / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) *
            (p11 ** 2 + p10 ** 2 - 2 * p11 * p10 * cosbeta2))

        # 目标函数
        objective = cp.Maximize(
            alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

        # 定义并求解问题
        problem = cp.Problem(objective, constraints)

        # 求解问题
        problem.solve(solver="MOSEK")  # SDPA OR mosek

        # 检查差值
        if abs(lambda1 - problem.value) < 1e-5:
            print(f"Found parameters: p00={p00:.2f}, p01={p01:.2f}, alpha={alpha:.2f}, "
                  f"lambda1={lambda1:.5f}, problem.value={problem.value:.5f}")


find_best_parameters()
