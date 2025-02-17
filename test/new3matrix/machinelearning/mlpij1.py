import numpy as np
from scipy.optimize import minimize

# 定义公式左侧的计算，p00, p01, p10, p11 固定为1
def left_side(b, t):
    p00 = p01 = p10 = p11 = 1  # 固定参数
    term1 = p10 * (p00 + p10 * b) / np.sqrt((p00 + p10 * b) ** 2 + p10 ** 2 * (1 - b ** 2) * (1 - t ** 2))
    term2 = p11 * (p01 - p11 * b) / np.sqrt((p01 - p11 * b) ** 2 + p11 ** 2 * (1 - b ** 2) * (1 - t ** 2))
    return term1 - term2

# 定义损失函数（目标是使得公式左侧尽量接近0）
def loss(params):
    b, t = params
    return left_side(b, t) ** 2  # 平方误差

# 执行优化
def optimize_bt():
    results = []  # 存储结果

    # 执行20次优化，使用不同的初始值
    for i in range(20):
        # 保证初始值在有效范围内
        x0 = np.array([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])  # 初始化 b 和 t 为随机值
        bounds = [(0.01, 0.99), (0.01, 0.99)]  # 约束b, t 在 (0.01, 0.99) 范围内

        # 使用 SLSQP 优化方法，优化 b 和 t
        result = minimize(loss, x0, bounds=bounds, method='SLSQP', options={'disp': False, 'maxiter': 500})

        # 获取优化结果
        b_opt, t_opt = result.x
        left_value = left_side(b_opt, t_opt)

        # 记录结果
        results.append((b_opt, t_opt, left_value))

        # 输出每次优化的结果
        print(f"第{i+1}组结果：")
        print(f"b = {b_opt}, t = {t_opt}")
        print(f"公式左侧值 = {left_value:.6e}")
        print("-" * 50)

    return results

# 执行优化并输出20组结果
optimize_bt()
