import numpy as np
from scipy.optimize import minimize, differential_evolution


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


# 约束条件：使 p00 = p01 = p10 = p11，或者相近
def equality_constraint(p):
    # 强制相等
    return p[0] - p[1], p[0] - p[2], p[0] - p[3]  # 强制 p00 = p01 = p10 = p11


# 正则化项：鼓励参数远离边界
def regularization(p):
    return np.sum(np.clip(p, 0.01, 0.99) - p) ** 2  # 惩罚偏离边界的程度


# 增强版损失函数：包括正则化项
def enhanced_loss(p):
    return loss(p) + 0.1 * regularization(p)  # 添加正则化项，权重可以调整


# 全局优化方法：使用 `differential_evolution`
def global_optimization():
    bounds = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95)]  # 边界范围

    # 运行全局优化算法
    result = differential_evolution(enhanced_loss, bounds)
    p_opt = result.x
    left_value = left_side(p_opt[:4], p_opt[4], p_opt[5])

    print("全局优化结果：")
    print(f"p00 = {p_opt[0]}, p01 = {p_opt[1]}, p10 = {p_opt[2]}, p11 = {p_opt[3]}")
    print(f"b = {p_opt[4]}, t = {p_opt[5]}")
    print(f"公式左侧值 = {left_value:.6e}")
    print("-" * 50)

    return result.x


# 使用多启动优化策略
def multi_start_optimization(num_starts=10):
    results = []
    for i in range(num_starts):
        x0 = np.random.rand(6) * 0.5 + 0.25  # 在 (0,1) 范围内随机初始化
        bounds = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95)]  # 修改边界

        # 使用优化算法进行最小化，设置约束条件
        constraints = [{'type': 'eq', 'fun': equality_constraint}]  # 强制 p00 = p01 = p10 = p11

        # 使用 trust-constr 算法
        result = minimize(enhanced_loss, x0=x0, bounds=bounds, constraints=constraints, method='trust-constr')

        # 记录优化后的结果
        p00_opt, p01_opt, p10_opt, p11_opt, b_opt, t_opt = result.x
        results.append(result.x)  # 将结果存入列表

        # 计算公式左侧值（验证优化效果）
        left_value = left_side([p00_opt, p01_opt, p10_opt, p11_opt], b_opt, t_opt)

        # 输出每次优化的结果（保留小数，输出参数和公式左侧值）
        print(f"启动 {i + 1}：")
        print(f"p00 = {p00_opt}, p01 = {p01_opt}, p10 = {p10_opt}, p11 = {p11_opt}")
        print(f"b = {b_opt}, t = {t_opt}")
        print(f"公式左侧值 = {left_value:.6e}")  # 使用科学计数法输出公式结果
        print("-" * 50)

    return results


# 执行全局优化
global_optimization()

# 执行多启动优化策略
results = multi_start_optimization(num_starts=10)

# 总结优化结果
print("\n优化结果总结：")
for i, res in enumerate(results):
    p00_opt, p01_opt, p10_opt, p11_opt, b_opt, t_opt = res
    left_value = left_side([p00_opt, p01_opt, p10_opt, p11_opt], b_opt, t_opt)
    print(f"启动 {i + 1} 最优参数：p00={p00_opt}, p01={p01_opt}, p10={p10_opt}, p11={p11_opt}, b={b_opt}, t={t_opt}")
    print(f"公式左侧值 = {left_value:.6e}")  # 使用科学计数法输出公式结果
