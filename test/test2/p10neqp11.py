import numpy as np
from sympy import symbols, Eq, solve, sin, cos, sqrt

# 初始化参数范围
step = 0.05
alpha_initial = 0.2
alpha_step = 0.2
max_value = 1

# 存储结果
results = []

# 遍历 p00, p01, p10, p11 的值
for p00 in np.arange(1, max_value + step, step):  # p00 初始值为 1
    for p01 in np.arange(0, max_value + step, step):
        for p10 in np.arange(1, max_value + step, step):
            for p11 in np.arange(0, max_value + step, step):
                if p10 == p11:
                    continue  # 确保 p10 不等于 p11

                # 遍历 alpha 的值
                for alpha in np.arange(alpha_initial, 2 + alpha_step, alpha_step):
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
                    eq1 = Eq((p10 * sin(beta2) * A / sqrt(A ** 2 + B ** 2)),
                             (p11 * sin(beta2) * C / sqrt(C ** 2 + D ** 2)))
                    eq2 = Eq(alpha, ((p10 ** 2 * sin(beta2) / sqrt(A ** 2 + B ** 2)) +
                                     (p11 ** 2 * sin(beta2) / sqrt(C ** 2 + D ** 2))) * cos(2 * theta))

                    # 解方程
                    solutions = solve((eq1, eq2), (beta2, theta))
                    for solution in solutions:
                        # 检查解的有效性
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

# 输出结果到文件
with open('results.txt', 'w') as f:
    for result in results:
        f.write(f"{result}\n")
