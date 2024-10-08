import numpy as np
import cvxpy as cp
import math
from functions import measure_pure_state, sort_numbers_with_names
from sympy import symbols, Eq, solve, cos, sin
from scipy.optimize import fsolve

# 此文件用于验证各种数值

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
for i in range(1):
    while True:
        values = np.random.uniform(0.0, 1.0, 4)
        alpha = np.random.uniform(0.0, 2.0)
        p00, p01, p10, p11 = sorted(values, reverse=True)  # 确保 p11 是最小的值
        # cos_beta = p00 * (math.sqrt(p11) - 1) / (p10 + p11 ** 2 * math.sqrt(p11))
        if p00 - (p10 + p11) < p01 < p00 + (p10 + p11) and alpha ** 2 < (
                (p11 ** 2) * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 + 2 * p01 * p11 * (p11 + p10) * (p01 - p00)) / (
                (p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2) < 1:
            break
        # TODO: if中的第一个条件是否应该存在

    # 检查变量关系是否满足条件
    # if 1 / p00 + 1 / p01 + 1 / p10 - 1 / p01 > 0:  # ?还是这个吗 还包含这个吗

    # 目标函数
    objective = cp.Maximize(
        alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

    # 定义并求解问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    problem.solve(solver="SDPA")  # SDPA OR mosek

    # F & cos(beta) TODO: may not correct
    F = p10 / p11 + 1
    cos_beta = (p01 - p00) / (p11 + p10)

    # F_2 = ((p00 + p10 * cos_beta) * p10) / ((p01 - p11 * cos_beta) * p11) + 1

    # 计算L_Q
    L_Q = F * math.sqrt(
        (1 + (alpha ** 2) / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p01 ** 2 - 2 * p11 * p01 * cos_beta))
    # L_Q2 = F_2 * math.sqrt(
    #     (1 + (alpha ** 2) / ((p11 + p10) ** 2 - (p01 - p00) ** 2)) * (p11 ** 2 + p01 ** 2 - 2 * p11 * p01 * cos_beta))
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

    # 计算θ的值
    theta = math.acos(cos_2theta) / 2

    # 计算 cos(θ) 和 sin(θ)
    cos_theta = np.sqrt((cos_2theta + 1) / 2)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    # 将θ的值转换为度数（可选）
    theta_degrees = math.degrees(theta)

    sin_beta1 = 0
    cos_beta1 = 1

    ########################################################################
    # 态和测量                                      #
    ########################################################################
    numerator1 = p00 + p10 * cos_beta
    denominator1 = math.sqrt((p00 + p10 * cos_beta) ** 2 + (p10 * sin_beta * sin_2theta) ** 2)
    cos_miu1 = numerator1 / denominator1

    numerator2 = p10 * sin_beta * sin_2theta
    denominator2 = math.sqrt((p00 + p10 * cos_beta) ** 2 + (p10 * sin_beta * sin_2theta) ** 2)
    sin_miu1 = numerator2 / denominator2

    numerator3 = p01 - p11 * cos_beta
    denominator3 = math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2)
    cos_miu2 = numerator3 / denominator3

    numerator4 = p11 * sin_beta * sin_2theta
    denominator4 = math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2)
    sin_miu2 = - numerator4 / denominator4  # TODO: not not correct

    # 定义量子态 |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
    psi = cos_theta * np.array([1, 0, 0, 0]) + sin_theta * np.array([0, 0, 0, 1])

    # 定义测量算符 A0, A1, B0, B1
    sigma_z = np.array([[1, 0], [0, -1]])  # σz
    sigma_x = np.array([[0, 1], [1, 0]])  # σx

    A0 = sigma_z
    A1 = cos_beta * sigma_z + sin_beta * sigma_x
    B0 = cos_miu1 * sigma_z + sin_miu1 * sigma_x  # cos(μ1)σz + sin(μ1)σx
    B1 = cos_miu2 * sigma_z + sin_miu2 * sigma_x  # cos(μ2)σz + sin(μ2)σx

    # 计算量子态的密度矩阵
    density_matrix = np.outer(psi, psi.conj())

    # 计算测量结果
    # A0_measurement = np.trace(density_matrix @ np.kron(A0, np.eye(2)))
    # A0B0_measurement = np.trace(density_matrix @ np.kron(A0, B0))
    # A0B1_measurement = np.trace(density_matrix @ np.kron(A0, B1))
    # A1B0_measurement = np.trace(density_matrix @ np.kron(A1, B0))
    # A1B1_measurement = np.trace(density_matrix @ np.kron(A1, B1))
    A0_measurement = measure_pure_state(psi, A0, np.eye(2))
    A0B0_measurement = measure_pure_state(psi, A0, B0)
    A0B1_measurement = measure_pure_state(psi, A0, B1)
    A1B0_measurement = measure_pure_state(psi, A1, B0)
    A1B1_measurement = measure_pure_state(psi, A1, B1)

    Iap = alpha * A0_measurement + p00 * A0B0_measurement + p01 * A0B1_measurement + \
          p10 * A1B0_measurement - p11 * A1B1_measurement


    # ----------------------------------------
    # 用 A13 & A5 get cos_beta & cos_theta
    def equation1(theta_1):
        return alpha * cos_beta1 - ((p10 * sin_miu1 - p11 * sin_miu2) * sin_beta * np.cos(2 * theta_1) / np.sin(
            2 * theta_1) +
                                    (p00 * sin_miu1 + p01 * sin_miu2) * sin_beta1 * np.cos(2 * theta_1) / np.sin(
                    2 * theta_1))


    # theta_1, beta_2 = symbols('theta_1 beta_2')
    # equation1 = Eq(alpha * cos_beta1 - ((p10 * sin_miu1 - p11 *
    # sin_miu2) * sin_beta * cos(2 * theta_1) / sin(2 * theta_1) + (p00 * sin_miu1 + p01 * sin_miu2) * sin_beta1 *
    # cos(2 * theta_1) / sin(2 * theta_1)), 0)
    # equation2 = Eq(p10 * sin(beta_2) * (p00 + p10 * cos(beta_2)) / ((p00 +
    # p10 * cos(beta_2))**2 + (p10 * sin(beta_2) * sin(2*theta_1))**2) - p11 * sin(beta_2) * (p01 - p11 * cos(
    # beta_2)) / ((p01 - p11 * cos(beta_2))**2 + (p11 * sin(beta_2) * sin(2*theta_1))**2), 0)
    #
    # solution = solve((equation1, equation2), (theta_1, beta_2))

    # 定义初始猜测值的范围
    initial_guesses = np.linspace(0.01, np.pi / 4 - 0.01, 10)

    # 使用fsolve求解
    solutions = []
    for guess in initial_guesses:
        solution = fsolve(equation1, guess)
        if 0 < solution[0] < np.pi / 4:
            solutions.append(solution[0])

    # 去重并计算 cos(θ)
    unique_solutions = np.unique(solutions)
    cos_theta_values = np.cos(unique_solutions)
    theta_1 = unique_solutions


    def equation2(beta_2):
        return p10 * np.sin(beta_2) * (p00 + p10 * np.cos(beta_2)) / ((p00 + p10 * np.cos(
            beta_2)) ** 2 + (p10 * np.sin(beta_2) * np.sin(2 * theta_1)) ** 2) - p11 * np.sin(beta_2) * (
                p01 - p11 * np.cos(
            beta_2)) / ((p01 - p11 * np.cos(beta_2)) ** 2 + (p11 * np.sin(beta_2) * np.sin(2 * theta_1)) ** 2)

    # 使用fsolve求解
    solutions = []
    for guess in initial_guesses:
        solution = fsolve(equation2, guess)
        if 0 < solution[0] < np.pi:
            solutions.append(solution[0])

    unique_solutions2 = np.unique(solutions)

    # ----------------------------------------
    lambda1 = F * math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2) + alpha * cos_2theta
    lambda2 = ((p00 * cos_beta1 + p10 * cos_beta) * cos_miu1 + (
            p01 * cos_beta1 - p11 * cos_beta) * cos_miu2) + (
                      (p00 * sin_beta1 + p10 * sin_beta) * sin_miu1 + (
                      p01 * sin_beta1 - p11 * sin_beta) * sin_miu2) * sin_2theta + alpha * cos_2theta * cos_beta1
    lambda3 = math.sqrt(  # A14-2
        (p00 * cos_beta1 + p10 * cos_beta) ** 2 + (p00 * sin_beta1 + p10 * sin_beta) ** 2 * (sin_2theta) ** 2) \
              + math.sqrt((p01 * cos_beta1 - p11 * cos_beta) ** 2 + (p01 * sin_beta1 - p11 * sin_beta) ** 2 * (
        sin_2theta) ** 2) + alpha * cos_2theta * cos_beta1
    lambda4 = math.sqrt(  # A14-3
        (p00 + p10 * cos_beta) ** 2 + (p10 * sin_beta * sin_2theta) ** 2) \
              + math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2) + alpha * cos_2theta
    # A14-4
    lambda5_1 = F * math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2) + alpha * cos_2theta
    # lambda5_2 = F_2 * math.sqrt((p01 - p11 * cos_beta) ** 2 + (p11 * sin_beta * sin_2theta) ** 2) + alpha * cos_2theta

    a12 = alpha * sin_beta1 + (p00 * cos_beta1 + p10 * cos_beta) * sin_miu1 + (
            p01 * cos_beta1 - p11 * cos_beta) * sin_miu2
    a13 = (p00 * sin_beta1 + p10 * sin_beta) * cos_miu1 + (p01 * sin_beta1 - p11 * sin_beta) * cos_miu2
    what = p10 * cos_miu1 - p11 * cos_miu2

    # TODO: which F? F_2?
    # ----------------------------------------
    # 输出结果
    print(f"Iteration {i + 1}:")
    print("p00:", p00)
    print("p01:", p01)
    print("p10:", p10)
    print("p11:", p11)
    print("F:", F)
    # print("F2:", F_2)
    print("----------------------------------------")

    # ----------------------------------------
    print("cos_beta:", cos_beta)
    print("alpha:", alpha)
    print(f"β的值为: {beta} 弧度")
    print(f"β的值为: {beta_degrees} 度")
    print("cos2theta:", cos_2theta)
    print("cosθ:", cos_theta)
    print(f"θ的值为: {theta} 弧度")
    print(f"θ的值为: {theta_degrees} 度")
    print("----------------------------------------")

    print(f"cos(μ1)的值为: {cos_miu1}")
    print(f"sin(μ1)的值为: {sin_miu1}")
    print(f"cos(μ2)的值为: {cos_miu2}")
    print(f"sin(μ2)的值为: {sin_miu2}")
    print("----------------------------------------")

    print("A0 测量结果:", A0_measurement)
    print("A0B0 测量结果:", A0B0_measurement)
    print("A0B1 测量结果:", A0B1_measurement)
    print("A1B0 测量结果:", A1B0_measurement)
    print("A1B1 测量结果:", A1B1_measurement)
    print("Iap = ", Iap)
    print("----------------------------------------")

    # 验证一些数值
    print(f"解出的 cos(θ) 值为: {cos_theta_values}")
    # print(solution)
    print("----------------------------------------")

    print("A0 in gamma", gamma[0, 1].value)
    print("A0B0 in gamma", gamma[1, 3].value)
    print("A0B1 in gamma", gamma[1, 4].value)
    print("A1B0 in gamma", gamma[2, 3].value)
    print("A1B1 in gamma", gamma[2, 4].value)
    print("lambda1:", lambda1)  # A20
    print("lambda2:", lambda2)  # A14-1
    print("lambda3:", lambda3)  # A14-2
    print("lambda4:", lambda4)  # A14-3
    print("lambda51:", lambda5_1)  # A14-4
    print("A12:", a12)
    print("A13:", a13)
    print("what:", what)
    # print("lambda52:", lambda5_2)  # A14-4
    print("----------------------------------------")
    # ----------------------------------------

    print("Optimal value:", problem.value)
    print("Optimal matrix X:", gamma.value)
    print("L_Q:", L_Q)
    # print("L_Q2:", L_Q2)
    print("Is 实际值大于理论值 ", problem.value > L_Q)  # ......?
    print("Difference:", problem.value - L_Q)  # IQ为理论值，value为实际值

    print("sorted values 从小到大:", sort_numbers_with_names(problem.value, L_Q, Iap))
    print("\n")
    # else:
    # print(f"Iteration {i + 1}: 条件不满足，跳过此迭代\n")