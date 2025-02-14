import numpy as np
from scipy.optimize import minimize


# 计算 sin(mu1), sin(mu2), cos(mu1), cos(mu2)
def compute_trig_functions(beta2, p00, p01, p10, p11, theta):
    cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu2 = - (p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return cos_mu1, cos_mu2, sin_mu1, sin_mu2


# 计算第一个公式左侧
def left_side_1(beta2, p00, p01, p10, p11, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    term1 = (p00 + p10 * np.cos(beta2)) * (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    term2 = (p01 - p11 * np.cos(beta2)) * (-p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return term1 + term2


# 计算第二个公式左侧
def left_side_2(beta2, p00, p01, p10, p11, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    term1 = p10 * np.sin(beta2) * (p00 + p10 * np.cos(beta2)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    term2 = -p11 * np.sin(beta2) * (p01 - p11 * np.cos(beta2)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return term1 + term2


# 目标函数：计算两个公式左侧的平方和
def objective(params, beta2, theta):
    p00, p01, p10, p11 = params
    # 计算两个公式的左侧
    left_1 = left_side_1(beta2, p00, p01, p10, p11, theta)
    left_2 = left_side_2(beta2, p00, p01, p10, p11, theta)
    # 返回两个公式左侧的平方和
    return left_1 ** 2 + left_2 ** 2


# 优化过程：使用最小化方法
def optimize_parameters(beta2, theta):
    # 初始参数猜测 (p00, p01, p10, p11)
    initial_params = [1.0, 0.5, 0.5, 0.3]

    # 最小化目标函数
    result = minimize(objective, initial_params, args=(beta2, theta),
                      bounds=[(None, None), (None, None), (None, None), (None, None)])

    return result.x  # 返回优化后的参数


# 示例参数
beta2 = np.pi / 4  # 例如 45 度
theta = np.pi / 6  # 例如 30 度

# 优化参数
optimized_params = optimize_parameters(beta2, theta)

print(f"Optimized p parameters: {optimized_params}")
