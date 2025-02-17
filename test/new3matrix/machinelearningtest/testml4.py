import numpy as np
from scipy.optimize import minimize

# 计算 A13 的函数，接受 p00, p01, p10, cosbeta2 和 cos2theta
def compute_A13(p00, p01, p10, cosbeta2, cos2theta):
    # 计算左边的表达式
    left_side = (p10 * (p00 + p10 * cosbeta2)) / np.sqrt((p00 + p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2)) \
               - (p10 * (p01 - p10 * cosbeta2)) / np.sqrt((p01 - p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2))
    return left_side

# 目标函数，最小化 compute_A13 值与 0 的差异
def objective(params):
    p00, p01, p10, cosbeta2, cos2theta = params
    return abs(compute_A13(p00, p01, p10, cosbeta2, cos2theta))  # Minimize the absolute difference from 0

# 优化约束：p00, p01, p10, cosbeta2, cos2theta 必须在 (0.1, 0.9) 范围内
bounds = [(0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9)]

# 生成10组解
num_samples = 10
optimized_results = []

for _ in range(num_samples):
    # 随机生成初始猜测，确保在(0.1, 0.9)范围内，避免接近边界
    initial_guess = [np.random.uniform(0.1, 0.9) for _ in range(5)]

    # 使用 minimize 进行优化，选择 trust-constr 算法并调整精度设置
    result = minimize(objective, initial_guess, method='trust-constr', bounds=bounds,
                      options={'disp': True, 'maxiter': 5000, 'gtol': 1e-6, 'xtol': 1e-6})

    # 获取优化结果
    optimized_p00, optimized_p01, optimized_p10, optimized_cosbeta2, optimized_cos2theta = result.x
    optimized_A13 = compute_A13(optimized_p00, optimized_p01, optimized_p10, optimized_cosbeta2, optimized_cos2theta)

    # 保存结果
    optimized_results.append((optimized_p00, optimized_p01, optimized_p10, optimized_cosbeta2, optimized_cos2theta, optimized_A13))

# 打印优化结果，不进行格式化
for i, (p00, p01, p10, cosbeta2, cos2theta, A13) in enumerate(optimized_results):
    print(f"Optimized Set {i+1}: p00 = {p00}, p01 = {p01}, p10 = {p10}, cosbeta2 = {cosbeta2}, cos2theta = {cos2theta}, A13 = {A13}")
