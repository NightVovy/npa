import numpy as np
from scipy.optimize import fsolve

# 运行5次
for i in range(5):
    print(f"运行第{i+1}次:")

    # 随机生成p00, p01, p10, p11的值
    p00, p01, p10, p11 = np.random.rand(4)

    # 定义方程组
    def equations(cosbeta):
        eq1 = p00 + p10 * cosbeta
        eq2 = p01 - p11 * cosbeta
        eq3 = p10 * (p01 - p11 * cosbeta)**3 + p11 * (p00 + p10 * cosbeta)**3
        return [eq1.item(), eq2.item(), eq3.item()]

    # 求解cosbeta
    cosbeta_solutions = fsolve(equations, [0])

    # 检查cosbeta是否在[-1, 1]范围内
    valid_cosbeta = [cosbeta for cosbeta in cosbeta_solutions if -1 <= cosbeta <= 1]

    # 如果没有有效的cosbeta，跳过
    if not valid_cosbeta:
        print("没有有效的cosbeta解")
    else:
        for cosbeta in valid_cosbeta:
            # 计算A和B
            A = (p00 + p10 * cosbeta)**2
            B = (p01 - p11 * cosbeta)**2

            # 计算sinbeta^2 / sin2theta^2
            sinbeta2_sin2theta2 = (p10**2 - p11**2) * A * B / (p10**2 * p11**2 * (B - A))

            # 计算sinbeta和theta
            sinbeta = np.sqrt(sinbeta2_sin2theta2)
            theta = np.arcsin(np.sqrt(sinbeta2_sin2theta2))

            # 计算lambda
            lambda_val = np.sqrt((p00 + p10 * cosbeta)**2 + (p10**2 * sinbeta**2 / sinbeta2_sin2theta2)) + \
                         np.sqrt((p01 - p11 * cosbeta)**2 + (p11**2 * sinbeta**2 / sinbeta2_sin2theta2))

            # 输出theta、cosbeta、sinbeta和lambda的值
            print(f"theta: {theta}, cosbeta: {cosbeta}, sinbeta: {sinbeta}, lambda: {lambda_val}")

    print()
