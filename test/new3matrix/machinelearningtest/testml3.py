import numpy as np
from scipy.optimize import minimize

# 计算 A13 的函数，只接受 cosbeta2 和 cos2theta
def compute_A13(cosbeta2, cos2theta):
    # 通过 arccos 获取 beta2
    beta2 = np.arccos(cosbeta2)

    # 计算左边的表达式
    left_side = (beta2 + cosbeta2) / np.sqrt((beta2 + cosbeta2)**2 + (1 - cosbeta2**2)*(1 - cos2theta**2)) \
               - (beta2 - cosbeta2) / np.sqrt((beta2 - cosbeta2)**2 + (1 - cosbeta2**2)*(1 - cos2theta**2))
    return left_side

# 目标函数，最小化 compute_A13 值与 0 的差异
def objective(params):
    cosbeta2, cos2theta = params
    return abs(compute_A13(cosbeta2, cos2theta))  # Minimize the absolute difference from 0

# 优化约束：cosbeta2, cos2theta 必须在 (0.1, 0.9) 范围内
bounds = [(0.1, 0.9), (0.1, 0.9)]

# 生成10组解
num_samples = 10
optimized_results = []

for _ in range(num_samples):
    # 随机生成初始猜测，确保在(0.1, 0.9)范围内，避免接近边界
    initial_guess = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]

    # 使用 minimize 进行优化，选择 trust-constr 算法并调整精度设置
    result = minimize(objective, initial_guess, method='trust-constr', bounds=bounds,
                      options={'disp': True, 'maxiter': 5000, 'gtol': 1e-6, 'xtol': 1e-6})

    # 获取优化结果
    optimized_cosbeta2, optimized_cos2theta = result.x
    optimized_A13 = compute_A13(optimized_cosbeta2, optimized_cos2theta)
    beta2 = np.arccos(optimized_cosbeta2)  # 计算 beta2

    # 保存结果
    optimized_results.append((beta2, optimized_cosbeta2, optimized_cos2theta, optimized_A13))

# 打印优化结果
for i, (beta2, cosbeta2, cos2theta, A13) in enumerate(optimized_results):
    print(f"Optimized Set {i+1}: beta2 = {beta2:.5f}, cosbeta2 = {cosbeta2:.5f}, cos2theta = {cos2theta:.5f}, A13 = {A13:.10f}")
