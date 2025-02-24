import numpy as np
from scipy.optimize import minimize


# 定义 left_side 公式
def left_side(params):
    p, cosbeta2, cos2theta = params
    term1 = p * (p + p * cosbeta2) / np.sqrt(
        (p + p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    term2 = p * (p - p * cosbeta2) / np.sqrt(
        (p - p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    return abs(term1 - term2)  # 绝对值最小化


# 生成多组初始值
solutions = []
while len(solutions) < 20:  # 确保有20组有效数据
    initial_guess = np.random.uniform(0.1, 0.9, 3)  # 取 (0.1, 0.9) 之间的中间值
    result = minimize(left_side, initial_guess, bounds=[(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)])

    if result.success:
        p_opt, cosbeta2_opt, cos2theta_opt = result.x
        left_side_value = left_side(result.x)

        # 只保留 p, cosbeta2, cos2theta 均不接近边界的数据
        if 0.1 < p_opt < 0.9 and 0.1 < cosbeta2_opt < 0.9 and 0.1 < cos2theta_opt < 0.9:
            solutions.append((p_opt, cosbeta2_opt, cos2theta_opt, left_side_value))

# 输出结果，确保不使用科学计数法
print("    p               cosbeta2         cos2theta         left_side")
print("-----------------------------------------------------------------")
for p, cosbeta2, cos2theta, left_side_value in solutions:
    print(f"{p:.10f}   {cosbeta2:.10f}   {cos2theta:.10f}   {left_side_value:.10f}")
