import math

import numpy as np
import cvxpy as cp

# 定义变量
gamma = cp.Variable((9, 9))
# gamma = cp.Variable((9, 9), complex=True)

# 定义CHSH函数
B = np.array([[0, -1, 0], [-1, 1, 1], [0, 1, -1]])

# 将B扩展到9x9矩阵
# B_expanded = np.zeros((9, 9))
# B_expanded[:3, :3] = B

# 定义目标函数
# objective = cp.Maximize(cp.trace(B_expanded.T @ gamma))
objective = cp.Maximize(gamma[1, 3] + gamma[1, 4] + gamma[2, 3] - gamma[2, 4])
# maximize ...

# 定义约束条件
constraints = [
    gamma >> 0,  # 半正定约束
    gamma == cp.conj(gamma.T),  # 共轭转置对称
    gamma[0, 0] == 1,
    cp.diag(gamma)[1:] == 1,  # 除了第一行第一列元素为0外，其他对角线元素为1

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

# 定义并求解问题
problem = cp.Problem(objective, constraints)
problem.solve(solver="SDPA")

# 输出结果
print("Optimal value:", problem.value)
print("Optimal gamma matrix:", gamma.value)
print("Is 理论值2sqrt2>实际值?:", 2 * math.sqrt(2) > problem.value)

# TODO: function product_of_orthogonal()
# TODO: index matrix
