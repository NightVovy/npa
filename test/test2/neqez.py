import numpy as np
from sympy import symbols, Eq, solve, sin, cos, sqrt

# 初始化参数范围
step = 0.2  # 更大的步长
max_value = 1

# 存储结果
results = []

# 遍历 p00, p01, p10, p11 的值
p00 = 1  # p00 固定为 1
for p01 in [0, 0.5, 1]:  # 限制 p01 的取值
    for p10 in [1]:  # 固定 p10
        for p11 in [0, 0.5]:  # 限制 p11 的取值
            if p10 == p11:
                continue  # 确保 p10 不等于 p11

            # 遍历 alpha 的值
            for alpha in [0.2, 1.0, 1.8]:  # 限制 alpha 的取值
                # 定义符号变量
                beta2, theta = symbols('beta2 theta')

                # 计算 A, B, C, D
                A = p00 + p10 * cos(beta2)
                B = p10 * sin(beta2) * sin(2 * theta)
                C = p01 - p11 * cos(beta2)
                D = p11 * sin(beta2) * sin(2 * theta)

                # 确保分母不为零
                if A ** 2 + B ** 2 == 0 or C ** 2 + D ** 2 == 0:
                    continue

                # 设定方程
                eq1 = Eq((p10 * sin(beta2) * A / sqrt(A ** 2 + B ** 2)), (p11 * sin(beta2) * C / sqrt(C ** 2 + D ** 2)))
                eq2 = Eq(alpha, ((p10 ** 2 * sin(beta2) / sqrt(A ** 2 + B ** 2)) +
                                 (p11 ** 2 * sin(beta2) / sqrt(C ** 2 + D ** 2))) * cos(2 * theta))

                # 使用近似方法求解 beta2 和 theta
                try:
                    solutions = solve((eq1, eq2), (beta2, theta))
                    for solution in solutions:
                        beta2_val, theta_val = solution
                        if beta2_val.is_real and theta_val.is_real:
                            cos_beta2 = cos(beta2_val)
                            cos_2theta = cos(2 * theta_val)

                            # 确保 cos(beta2) 和 cos(2*theta) 在合法范围内
                            if abs(cos_beta2) < 1 and abs(cos_2theta) < 1:
                                # 计算 lambda1
                                lambda1 = ((p10 ** 2 * sin(beta2_val) / sqrt(A ** 2 + B ** 2)) +
                                           (p11 ** 2 * sin(beta2_val) / sqrt(C ** 2 + D ** 2))) * cos_2theta

                                # 保存结果
                                results.append((p00, p01, p10, p11, alpha, cos_beta2, cos_2theta, lambda1))
                except Exception as e:
                    print(f"Error solving for p01={p01}, p10={p10}, p11={p11}, alpha={alpha}: {e}")

# 打印结果
for result in results:
    print(result)
