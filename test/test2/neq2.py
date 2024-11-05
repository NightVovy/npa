import numpy as np
import cvxpy as cp
import math

def read_parameters(file_path):
    """ 从文件读取参数值 """
    parameters = []
    with open(file_path, 'r') as file:
        for line in file:
            param_dict = {}
            # 解析一行中的参数
            for item in line.strip().split(', '):
                key, value = item.split(': ')
                param_dict[key] = float(value)
            parameters.append(param_dict)
    return parameters

def find_best_parameters():
    # 从文件读取参数
    params_list = read_parameters('data2.txt')
    results = []  # 存储结果

    for params in params_list:
        p00 = params['p00']
        p01 = params['p01']
        p10 = params['p10']
        p11 = params['p11']
        alpha = params['alpha']
        beta = params['beta']
        theta = params['theta']

        # 计算 sin 和 cos 值
        cosbeta = math.cos(beta)
        sinbeta = math.sin(beta)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # 计算 lambda
        lambda1 = (math.sqrt((p00 + p10 * cosbeta) ** 2 + p10 ** 2 * sinbeta ** 2 * (2 * sintheta ** 2))) + \
                   (math.sqrt((p01 - p11 * cosbeta) ** 2 + p11 ** 2 * sinbeta ** 2 * (2 * sintheta ** 2))) + \
                   (alpha * (costheta ** 2))

        # 定义 gamma 变量
        gamma = cp.Variable((9, 9))

        # 定义约束条件
        constraints = [
            gamma >> 0,  # 半正定约束
            cp.diag(gamma) == 1,  # 对角线元素为1
            gamma == cp.conj(gamma.T),  # 共轭转置对称
            # 其他约束保持不变...
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
            alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4]
        )

        # 定义并求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SDPA")  # 使用 SDPA 求解

        # 存储结果
        results.append((p00, p01, alpha, beta, theta, lambda1, problem.value))

        # 打印最优矩阵 X
        print("Optimal matrix X:", gamma.value)

    # 输出所有结果
    print("参数组合结果：")
    for p00, p01, alpha, beta, theta, lambda1, sdp_value in results:
        print(f"p00={p00:.2f}, p01={p01:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, theta={theta:.2f}, "
              f"lambda={lambda1:.5f}, SDP value={sdp_value:.5f}")

find_best_parameters()
