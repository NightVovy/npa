# lambda和npa之间的差值
import numpy as np
import cvxpy as cp
import math

def find_best_parameters():
    # 定义初始参数
    p00_start = 0  # 不等于 0
    p01_start = 0  # 不等于 0
    p10_start = 1  # 固定 p10 值
    p11_start = 1  # 固定 p11 值
    step = 0.1  # 步长增加为 0.1
    tolerance = 1e-5

    # 创建所有可能的 p00、p01 组合
    p_values = np.arange(0.1, 1, step)  # p00 和 p01 不等于 0 和 1
    # 遍历 alpha
    alpha_values = np.arange(0.1, 2.1, 0.2)  # alpha 从 0.1 到 2，步长为 0.2

    results = []  # 存储符合条件的结果

    for alpha in alpha_values:
        for p00 in p_values:
            for p01 in p_values:
                if p00 == p01:  # 跳过 p00 = p01 的情况
                    continue

                # 固定 p10 和 p11
                p10 = p10_start
                p11 = p11_start

                # 计算 cosbeta2
                cosbeta2 = (p01 - p00) / (p11 + p10)
                if abs(cosbeta2) >= 1:
                    continue  # 确保 |cosbeta2| < 1

                # 计算 F 和 lambda1
                F = (((p00 + p10 * cosbeta2) * p10) / ((p01 - p11 * cosbeta2) * p11)) + 1
                lambda1 = F * math.sqrt(
                    (1 + alpha ** 2 / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p01 ** 2 - 2 * p11 * p01 * cosbeta2))

                # 计算 alpha2
                alpha2 = (p11 ** 2 * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 + 2 * p01 * p11 * (p11 + p10) * (p01 - p00)) / (
                    (p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2)

                # 确保 alpha^2 < alpha2
                if alpha ** 2 >= alpha2:
                    continue  # 跳过不满足条件的 alpha

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

                # 直接输出 SDP 和 lambda 之间的差值
                print(f"p00={p00:.2f}, p01={p01:.2f}, alpha={alpha:.2f}, "
                      f"lambda={lambda1:.5f}, SDP={problem.value:.5f}, "
                      f"差值={abs(lambda1 - problem.value):.5e}")

find_best_parameters()