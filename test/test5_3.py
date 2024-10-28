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

# 定义alpha值[0,2]
alpha = 0.1

# 定义参数
# TODO: Pij的取值范围 [0,1]
p00 = 0.25
p01 = 0.25
p10 = 0.25
p11 = 0.25

# TODO： IS THIS CORRECT?
cosbeta2 = (p01 - p00) / (p11 + p10)

F = (((p00 + p10 * cosbeta2) * p10) / ((p01 - p11 * cosbeta2) * p11)) + 1
F2 = p10 / p11 + 1

# lambda1 = F * math.sqrt(
#     (1 + alpha ** 2 / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p10 ** 2 - 2 * p11 * p10 * np.cos(beta2)))

lambda1 = F * math.sqrt(
    (1 + alpha ** 2 / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p10 ** 2 - 2 * p11 * p10 * cosbeta2))

alpha2 = ((p11 ** 2) * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 + 2 * p01 * p11 * (p11 + p10) * (p01 - p00)) / (
        (p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2)

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

print("cosbeta2:", cosbeta2)
print("is |cosbeta2| < 1?:", abs(cosbeta2) < 1)
print("alpha^2:", alpha ** 2)
print("alpha constraint:", alpha2)
print("F:", F)
print("F2:", F2)
print("is alpha in constraint?:", alpha ** 2 < alpha2)
print("Result1 of lambda1:", lambda1)

print("和npa的差值1:", lambda1 - problem.value)
