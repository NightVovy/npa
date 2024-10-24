import numpy as np
import cvxpy as cp
import math

# 定义变量
gamma = cp.Variable((9, 9))

# 定义约束条件
constraints = [
    gamma >> 0,  # 半正定约束
    gamma == cp.conj(gamma.T),  # 共轭转置对称
    cp.diag(gamma) == 1,  # 对角线元素为1

    # 1/p00 + 1/p01 + 1/p10 - 1/p01 >= 0,  # 变量关系 ? 和下面冲突了
    # 对应同一个测量M的投影应满足：
    # 1. 正交 2. 求和为I 3.平方和共轭转置都是自身 4. Alice&Bob 方的投影算子对易[Ea,Eb]=0
    # Q：测量平方等于I？
    # these are not projectors

    gamma[0, 1] == gamma[3, 5],  # A0
    gamma[0, 1] == gamma[4, 6],

    gamma[0, 2] == gamma[3, 7],  # A1
    gamma[0, 2] == gamma[4, 8],

    gamma[0, 3] == gamma[1, 5],  # 第一行第四个元素等于第2行第六个元素 B0
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

# 定义beta值[0, pi/2 = 1.57]
beta1 = 1.2
beta2 = 1.4

# 定义alpha值[0,2]
alpha = 1

# 定义参数
p00 = beta1
p01 = beta2
p10 = 1
p11 = 1

# 目标函数
objective = cp.Maximize(
    alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

# 定义并求解问题
problem = cp.Problem(objective, constraints)

# 求解问题
problem.solve(solver="SDPA")  # SDPA OR mosek

# 输出结果
print("Optimal gamma:", gamma.value)
print("Optimal value:", problem.value)


# 计算 sqrt((4 + alpha ** 2) * (1 + beta ** 2)) 并输出
result1 = math.sqrt((4 + alpha ** 2) * (1 + beta1 ** 2))
result2 = math.sqrt((4 + alpha ** 2) * (1 + beta2 ** 2))
print("Result1 of sqrt((4 + alpha ** 2) * (1 + beta ** 2)):", result1)
print("差值1:", result1 - problem.value)
print("Result2 of sqrt((4 + alpha ** 2) * (1 + beta ** 2)):", result2)
print("差值2:", result2 - problem.value)
