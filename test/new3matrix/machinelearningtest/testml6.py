import numpy as np
from scipy.optimize import minimize

# 计算 alpha
def compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta):
    alpha = (p10**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p00 + p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2)) \
           + (p11**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p01 - p11 * cosbeta2)**2 + p11**2 * (1 - cosbeta2**2) * (1 - cos2theta**2))
    return alpha

# 计算公式 A13
def compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta):
    left_side = (p10 * (p00 + p10 * cosbeta2)) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + p10 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2)) \
                - (p11 * (p01 - p11 * cosbeta2)) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + p11 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    return left_side

# 计算公式 A14
def calculate_A14(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)
    sin2theta = np.sqrt(1 - cos2theta ** 2)

    term1 = np.sqrt((p00 + p10 * cosbeta2)**2 + (p10 * sinbeta2 * sin2theta)**2)
    term2 = np.sqrt((p01 - p11 * cosbeta2)**2 + (p11 * sinbeta2 * sin2theta)**2)
    term3 = alpha * cos2theta

    return term1 + term2 + term3

# 计算 Ilhv
def calculate_ilhv(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    return alpha + p00 + p01 + p10 - p11

# 损失函数（加大惩罚项）
def loss_function(params):
    p00, p01, p10, p11, cosbeta2, cos2theta = params
    alpha = compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta)
    A13_value = compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta)
    A14_value = calculate_A14(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)
    Ilhv_value = calculate_ilhv(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)

    # 更强的惩罚项：如果 A14 > Ilhv，则加大惩罚
    penalty = max(0, A14_value - Ilhv_value) ** 4  # 增强惩罚力度，惩罚与差值的四次方成正比
    return np.abs(A13_value) + penalty

# 设置约束条件
def constraints():
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.01},  # p00 > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[0]},  # p00 < 0.99
        {'type': 'ineq', 'fun': lambda x: x[1] - 0.01},  # p01 > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[1]},  # p01 < 0.99
        {'type': 'ineq', 'fun': lambda x: x[2] - 0.01},  # p10 > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[2]},  # p10 < 0.99
        {'type': 'ineq', 'fun': lambda x: x[3] - 0.01},  # p11 > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[3]},  # p11 < 0.99
        {'type': 'ineq', 'fun': lambda x: np.abs(x[2] - x[3]) - 0.001},  # p10 != p11
        {'type': 'ineq', 'fun': lambda x: x[4] - 0.01},  # cosbeta2 > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[4]},  # cosbeta2 < 0.99
        {'type': 'ineq', 'fun': lambda x: x[5] - 0.01},  # cos2theta > 0.01
        {'type': 'ineq', 'fun': lambda x: 0.99 - x[5]},  # cos2theta < 0.99
        {'type': 'ineq', 'fun': lambda x: calculate_A14(x[0], x[1], x[2], x[3], x[4], x[5], compute_alpha(x[0], x[1], x[2], x[3], x[4], x[5])) - calculate_ilhv(x[0], x[1], x[2], x[3], x[4], x[5], compute_alpha(x[0], x[1], x[2], x[3], x[4], x[5]))}  # A14 < Ilhv
    ]
    return cons

# 生成数据
def generate_data(num_samples=30):
    data = []
    while len(data) < num_samples:
        # 随机初始化参数，保证初始化在(0.01, 0.99)范围内
        initial_params = np.random.rand(6) * 0.98 + 0.01  # 保证初始化在(0.01, 0.99)范围内

        # 优化目标
        result = minimize(loss_function, initial_params, constraints=constraints(), method='SLSQP',
                          options={'disp': False})

        # 提取优化后的参数
        optimized_params = result.x
        p00, p01, p10, p11, cosbeta2, cos2theta = optimized_params

        # 检查是否有参数等于0.01或0.99，若有则忽略该组数据
        if np.any(np.isclose(optimized_params, 0.01)) or np.any(np.isclose(optimized_params, 0.99)):
            continue

        A13_value = compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta)
        A14_value = calculate_A14(p00, p01, p10, p11, cosbeta2, cos2theta, compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta))
        Ilhv_value = calculate_ilhv(p00, p01, p10, p11, cosbeta2, cos2theta, compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta))

        # 存储数据
        data.append([p00, p01, p10, p11, cosbeta2, cos2theta, A13_value, A14_value, Ilhv_value])

    return data

# 生成30组数据
data = generate_data(30)

# 输出每组数据
for i, params in enumerate(data):
    p00, p01, p10, p11, cosbeta2, cos2theta, A13_value, A14_value, Ilhv_value = params
    print(f"Group {i + 1}:")
    print(f"p00 = {p00:.15f}, p01 = {p01:.15f}, p10 = {p10:.15f}, p11 = {p11:.15f}, cosbeta2 = {cosbeta2:.15f}, cos2theta = {cos2theta:.15f}, A13 = {A13_value:.15f}, A14 = {A14_value:.15f}, Ilhv = {Ilhv_value:.15f}")
    print("-" * 80)
