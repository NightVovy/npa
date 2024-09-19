import numpy as np
import cvxpy as cp
import math

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
        if p00 - p10 + p11 < p01 < p00 + p10 + p11 and alpha ** 2 < (
                (p11 ** 2) * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 + 2 * p01 * p11 * (p11 + p10) * (p01 - p00)) / (
                (p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2):
            break

    # 检查变量关系是否满足条件
    # if 1 / p00 + 1 / p01 + 1 / p10 - 1 / p01 > 0:  # ?还是这个吗 还包含这个吗

    # 目标函数
    objective = cp.Maximize(
        alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

    # 定义并求解问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    problem.solve(solver="SDPA")

    # F & cos(beta)
    F = p10 / p11 + 1
    cos_beta = (p01 - p00) / (p11 + p10)

    # 计算L_Q
    L_Q = F * math.sqrt(
        (1 + (alpha ** 2) / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p01 ** 2 - 2 * p11 * p01 * cos_beta))

    # ----------------------------------------

    # beta
    beta = math.acos(cos_beta)
    sin_beta = math.sqrt(1 - cos_beta ** 2)

    # 将β的值转换为度数（可选）
    beta_degrees = math.degrees(beta)

    # cos2theta
    cos_2theta = alpha * math.sqrt(p01 ** 2 + p11 ** 2 - 2 * p11 * p01 * cos_beta) / \
                 (p11 * math.sqrt(1 - cos_beta ** 2) * math.sqrt(alpha ** 2 + (1 - cos_beta ** 2) * ((p11 + p10) ** 2)))
    sin_2theta = math.sqrt(1 - cos_2theta ** 2)

    # theta_2
    theta_2 = math.acos(cos_2theta)

    # 计算θ的值
    theta = theta_2 / 2

    # 将θ的值转换为度数（可选）
    theta_degrees = math.degrees(theta)

    # ----------------------------------------
    # 测量
    numerator1 = p00 + p10 * cos_beta
    denominator1 = math.sqrt((p00 + p10 * cos_beta) ** 2 + (p10 * sin_beta * sin_2theta) ** 2)
    cos_mu1 = numerator1 / denominator1

    numerator2 = p10 * sin_beta * sin_2theta
    denominator2 = math.sqrt((p00 + p10 * cos_beta) ** 2 + (p10 * sin_beta * sin_2theta) ** 2)
    sin_mu1 = numerator2 / denominator2

    numerator3 = p01 - p11 * cos_beta
    denominator3 = math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2)
    cos_mu2 = numerator3 / denominator3

    numerator4 = p11 * sin_beta * sin_2theta
    denominator4 = math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2)
    sin_mu2 = - numerator4 / denominator4

    # ----------------------------------------
    # 输出结果
    print(f"Iteration {i + 1}:")
    print("p00:", p00)
    print("p01:", p01)
    print("p10:", p10)
    print("p11:", p11)
    print("F:", F)
    print("----------------------------------------")

    # ----------------------------------------
    print("cos_beta:", cos_beta)
    print("alpha:", alpha)
    print(f"β的值为: {beta} 弧度")
    print(f"β的值为: {beta_degrees} 度")
    print("cos2theta:", cos_2theta)
    print(f"θ的值为: {theta} 弧度")
    print(f"θ的值为: {theta_degrees} 度")
    print("----------------------------------------")

    print(f"cos(μ1)的值为: {cos_mu1}")
    print(f"sin(μ1)的值为: {sin_mu1}")
    print(f"cos(μ2)的值为: {cos_mu2}")
    print(f"sin(μ2)的值为: {sin_mu2}")
    print("----------------------------------------")
    
    # ----------------------------------------

    print("Optimal value:", problem.value)
    print("Optimal matrix X:", gamma.value)
    print("L_Q:", L_Q)
    print("Is 实际值大于理论值 ", problem.value > L_Q)  # ......?
    print("Difference:", problem.value - L_Q)  # IQ为理论值，value为实际值
    print("\n")
    # else:
    # print(f"Iteration {i + 1}: 条件不满足，跳过此迭代\n")
