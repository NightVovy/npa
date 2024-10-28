import numpy as np
import cvxpy as cp
import math


def find_best_parameters():
    # 定义初始参数
    alpha = 1
    p00_start = 1  # 初始 p00 值
    p01_start = 1  # 初始 p01 值
    step = 0.05
    tolerance = 1e-5

    # 创建所有可能的 p00 和 p01 组合
    p_values = np.arange(0, 1 + step, step)
    for p00 in p_values:
        for p01 in p_values:
            if p00 + p01 > 1:
                continue  # 确保 p00 + p01 <= 1

            p10 = p11 = 1  # 让 p10 = p11，且保持它们的值相等

            # 计算 cosbeta2
            cosbeta2 = (p01 - p00) / (p11 + p10)
            if abs(cosbeta2) >= 1:
                continue  # 确保 |cosbeta2| < 1

            # 计算 F 和 lambda1
            F = (((p00 + p10 * cosbeta2) * p10) / ((p01 - p11 * cosbeta2) * p11)) + 1
            lambda1 = F * math.sqrt(
                (1 + alpha ** 2 / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (
                            p11 ** 2 + p10 ** 2 - 2 * p11 * p10 * cosbeta2))

            # 定义 gamma 变量
            gamma = cp.Variable((9, 9))

            # 定义约束条件
            constraints = [
                gamma >> 0,  # 半正定约束
                gamma == cp.conj(gamma.T),  # 共轭转置对称
                cp.diag(gamma) == 1,  # 对角线元素为1

                gamma[0, 1] == gamma[3, 5],
                gamma[0, 1] == gamma[4, 6],
                gamma[0, 2] == gamma[3, 7],
                gamma[0, 2] == gamma[4, 8],
                gamma[0, 3] == gamma[1, 5],
                gamma[0, 3] == gamma[2, 7],
                gamma[0, 4] == gamma[2, 8],
                gamma[0, 4] == gamma[1, 6],
                gamma[1, 3] == gamma[0, 5],
                gamma[1, 4] == gamma[0, 6],
                gamma[2, 3] == gamma[0, 7],
                gamma[2, 4] == gamma[0, 8],
                gamma[1, 2] == gamma[5, 7],
                gamma[1, 2] == gamma[6, 8],
                gamma[3, 4] == gamma[5, 6],
                gamma[3, 4] == gamma[7, 8],
                gamma[2, 5] == gamma[1, 7],
                gamma[2, 6] == gamma[1, 8],
                gamma[4, 5] == gamma[3, 6],
                gamma[3, 8] == gamma[4, 7],
                gamma[5, 7] == gamma[6, 8],
            ]

            # 定义目标函数
            objective = cp.Maximize(
                alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

            # 定义并求解问题
            problem = cp.Problem(objective, constraints)
            problem.solve(solver="SDPA")  # 使用 SDPA 求解

            # 检查差值
            if abs(lambda1 - problem.value) < tolerance:
                print(
                    f"找到合适的参数: p00={p00:.2f}, p01={p01:.2f}, lambda1={lambda1:.5f}, problem.value={problem.value:.5f}")
                return

    print("未找到满足条件的 p00 和 p01 组合。")


find_best_parameters()
