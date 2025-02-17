import numpy as np
from scipy.optimize import minimize


# 计算 left_side 的函数
def left_side(cosbeta2, cos2theta):
    term1 = (1 + cosbeta2) / np.sqrt((1 + cosbeta2) ** 2 + (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    term2 = (1 - cosbeta2) / np.sqrt((1 - cosbeta2) ** 2 + (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    return term1 - term2


# 目标函数，最小化 left_side 值与 0 的差异
def objective(params):
    cosbeta2, cos2theta = params
    return abs(left_side(cosbeta2, cos2theta))  # Minimize the absolute difference from 0


# 生成 10 组优化数据
num_samples = 10
optimized_results = []

for _ in range(num_samples):
    # 随机生成初始猜测，确保在(0.1, 0.9)范围内
    initial_guess = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]

    # 优化约束：cosbeta2, cos2theta 必须在 (0.1, 0.9) 范围内
    result = minimize(objective, initial_guess, bounds=[(0.1, 0.9), (0.1, 0.9)])

    # 获取优化结果
    optimized_cosbeta2, optimized_cos2theta = result.x
    optimized_results.append(
        (optimized_cosbeta2, optimized_cos2theta, left_side(optimized_cosbeta2, optimized_cos2theta)))

# 打印优化结果
for i, (cosbeta2, cos2theta, left_value) in enumerate(optimized_results):
    print(
        f"Optimized Set {i + 1}: cosbeta2 = {cosbeta2:.5f}, cos2theta = {cos2theta:.5f}, left_side = {left_value:.10f}")
