import numpy as np
from scipy.optimize import minimize


# 定义公式左侧的计算
def left_side(p, b, t):
    p00, p01, p10, p11 = p
    term1 = p10 * (p00 + p10 * b) / np.sqrt((p00 + p10 * b) ** 2 + p10 ** 2 * (1 - b ** 2) * (1 - t ** 2))
    term2 = p11 * (p01 - p11 * b) / np.sqrt((p01 - p11 * b) ** 2 + p11 ** 2 * (1 - b ** 2) * (1 - t ** 2))
    return term1 - term2


# 定义损失函数（目标是使得公式左侧尽量接近0）
def loss(p):
    p00, p01, p10, p11, b, t = p
    return left_side([p00, p01, p10, p11], b, t) ** 2  # 平方误差


# 多启动优化策略：多次随机初始化
def multi_start_optimization(num_starts=10):
    results = []

    # 运行多个优化，每次使用不同的随机初始值
    for i in range(num_starts):
        # 随机初始化参数
        x0 = np.random.rand(6) * 0.5 + 0.25  # 在 (0,1) 范围内随机初始化
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (-0.99, 0.99), (0.01, 0.99)]

        # 使用优化算法进行最小化
        result = minimize(loss, x0=x0, bounds=bounds)

        # 记录优化后的结果
        p00_opt, p01_opt, p10_opt, p11_opt, b_opt, t_opt = result.x
        results.append(result.x)  # 将结果存入列表

        # 计算公式左侧值（验证优化效果）
        left_value = left_side([p00_opt, p01_opt, p10_opt, p11_opt], b_opt, t_opt)

        # 输出每次优化的结果（保留小数，输出参数和公式左侧值）
        print(f"启动 {i + 1}：")
        print(f"p00 = {p00_opt}, p01 = {p01_opt}, p10 = {p10_opt}, p11 = {p11_opt}")
        print(f"cosbeta2 = {b_opt}, cos2theta = {t_opt}")
        print(f"公式左侧值 = {left_value:.6e}")  # 使用科学计数法输出公式结果
        print("-" * 50)

    return results


# 执行多启动优化策略，运行10次
results = multi_start_optimization(num_starts=10)

# 可以进一步分析不同启动的结果
print("\n优化结果总结：")
for i, res in enumerate(results):
    p00_opt, p01_opt, p10_opt, p11_opt, b_opt, t_opt = res
    left_value = left_side([p00_opt, p01_opt, p10_opt, p11_opt], b_opt, t_opt)
    print(f"启动 {i + 1} 最优参数：p00={p00_opt}, p01={p01_opt}, p10={p10_opt}, p11={p11_opt}, cosbeta2={b_opt}, cos2theta={t_opt}")
    print(f"公式左侧值 = {left_value:.6e}")  # 使用科学计数法输出公式结果
