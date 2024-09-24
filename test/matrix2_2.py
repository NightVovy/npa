import numpy as np
import cvxpy as cp
import math

# alpha = 0
# 定义变量
gamma = cp.Variable((9, 9))
# gamma = cp.Variable((9, 9), complex=True)

# p00 = cp.Variable()
# p01 = cp.Variable()
# p10 = cp.Variable()
# p11 = cp.Variable()


# 定义目标函数
# objective = cp.Maximize(p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])
# maximize ...

# 定义约束条件
constraints = [
    gamma >> 0,  # 半正定约束
    gamma == cp.conj(gamma.T),  # 共轭转置对称
    cp.diag(gamma) == 1,  # 对角线元素为1

    # 1/p00 + 1/p01 + 1/p10 - 1/p01 >= 0,  # 变量关系 ? 和下面冲突了
    # 对应同一个测量M的投影应满足：
    # 1. 正交 2. 求和为I 3.平方和共轭转置都是自身 4. Alice&Bob 方的投影算子对易[Ea,Eb]=0
    # Q：测量平方等于I？


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
# problem = cp.Problem(objective, constraints)

# 生成随机初始值并按步长赋值
np.random.seed(42)  # 固定随机种子以便复现结果
for i in range(10):
    while True:
        values = np.random.uniform(0.0, 1.0, 4)
        alpha = np.random.uniform(0.0, 2.0)
        p00, p01, p10, p11 = sorted(values, reverse=True)  # 确保 p11 是最小的值
        if 1/p00 + 1/p01 + 1/p10 - 1/p01 > 0:
            break


    # 目标函数
    objective = cp.Maximize(p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

    # 定义并求解问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    problem.solve(solver="SDPA")
    # problem.solve(solver="MOSEK")

    # 计算L_Q
    L_Q = math.sqrt((p00 * p10 + p01 * p11) * ((p00**2 + p10**2) / (p00 * p10) + (p01**2 + p11**2) / (p01 * p11)))

    # ----------------------------------------
    cos_theta = np.cos(np.pi / 4)  # 例如 θ = π/4
    sin_theta = np.sin(np.pi / 4)

    cos_beta = (p00 ** 2 * p10 ** 2 * (p01**2 + p11**2) - p01 ** 2 * p11 ** 2 * (p00**2 + p10**2))/ \
               (2 * p00*p01*p10*p11*(p00*p10+p01*p11))
    sin_beta = math.sqrt(1-cos_beta**2)

    alpha = 0

    # 定义量子态 |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
    psi = cos_theta * np.array([1, 0, 0, 0]) + sin_theta * np.array([0, 0, 0, 1])

    # 定义测量算符 A0, A1, B0, B1
    sigma_z = np.array([[1, 0], [0, -1]])  # σz
    sigma_x = np.array([[0, 1], [1, 0]])  # σx

    A0 = sigma_z
    A1 = sigma_x
    B0 = cos_miu1 * sigma_z + sin_miu1 * sigma_x  # cos(μ1)σz + sin(μ1)σx
    B1 = cos_miu2 * sigma_z + sin_miu2 * sigma_x  # cos(μ2)σz + sin(μ2)σx


    # 输出结果
    print(f"Iteration {i + 1}:")
    print("p00:", p00)
    print("p01:", p01)
    print("p10:", p10)
    print("p11:", p11)
    print("Optimal value:", problem.value)
    print("Optimal matrix X:", gamma.value)
    print("L_Q:", L_Q)
    print("Is actual result > Theoretical ", problem.value > L_Q)  # ......?
    print("Difference:", problem.value - L_Q)  # IQ为理论值，value为实际值
    print("\n")

