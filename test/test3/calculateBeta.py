import numpy as np
from scipy.optimize import root_scalar

def calculate_beta2(p, q, theta, epsilon=1e-8):
    """
    计算 beta2, cos(beta2), sin(beta2), cos(mu1), cos(mu2)
    确保除数不为 0 且 beta2 的解不为 0 或 pi
    """
    # 定义 p00, p01, p10, p11
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # 定义 beta2 方程
    def beta2_equation(beta2):
        sin_beta2 = np.sin(beta2)
        cos_beta2 = np.cos(beta2)

        # 计算分子和分母
        numerator1 = p00 + p10 * cos_beta2
        denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator1 < epsilon:
            print(f"Denominator1 is too small: {denominator1}")
        denominator1 = max(denominator1, epsilon)  # 确保分母不为 0
        cos_mu1 = numerator1 / denominator1

        numerator2 = p01 - p11 * cos_beta2
        denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator2 < epsilon:
            print(f"Denominator2 is too small: {denominator2}")
        denominator2 = max(denominator2, epsilon)  # 确保分母不为 0
        cos_mu2 = numerator2 / denominator2

        # 返回方程的结果
        return p10 * sin_beta2 * cos_mu1 - p11 * sin_beta2 * cos_mu2

    # 使用数值方法求解 beta2，排除 0 和 pi
    result = root_scalar(beta2_equation, bracket=[epsilon, np.pi - epsilon], method='brentq')

    if result.converged:
        beta2 = result.root
        cos_beta2 = np.cos(beta2)
        sin_beta2 = np.sin(beta2)

        # 最终计算 cos(mu1) 和 cos(mu2)
        numerator1 = p00 + p10 * cos_beta2
        denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
        print(f"Final numerator1: {numerator1}, denominator1: {denominator1}")
        denominator1 = max(denominator1, epsilon)  # 确保分母不为 0
        cos_mu1 = numerator1 / denominator1

        numerator2 = p01 - p11 * cos_beta2
        denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
        print(f"Final numerator2: {numerator2}, denominator2: {denominator2}")
        denominator2 = max(denominator2, epsilon)  # 确保分母不为 0
        cos_mu2 = numerator2 / denominator2

        # 计算 alpha
        sin_mu1 = np.sqrt(1 - cos_mu1**2)  # 计算 sin(mu1)
        sin_mu2 = np.sqrt(1 - cos_mu2**2)  # 计算 sin(mu2)

        alpha = (p10 * sin_mu1 - p11 * sin_mu2) * sin_beta2 * (np.cos(2 * theta) / np.sin(2 * theta))

        # 检查 alpha 是否在 (0, 2) 之间
        if not (0 < alpha < 2):
            print(f"Error: alpha 的值 {alpha} 不在 (0, 2) 之间。")

        return beta2, cos_beta2, sin_beta2, cos_mu1, cos_mu2, alpha
    else:
        print("Error: 无法找到 beta2 的解")
        return None

# 给定参数
# theta范围在(0,0.78539816339)
p = 0.1
q = 0.6
theta = 0.1

# 计算 beta2, cos(beta2), sin(beta2), cos(mu1), cos(mu2), alpha
result = calculate_beta2(p, q, theta)

if result:
    beta2, cos_beta2, sin_beta2, cos_mu1, cos_mu2, alpha = result

    # 输出结果
    print(f"beta2: {beta2}")
    print(f"cos(beta2): {cos_beta2}")
    print(f"sin(beta2): {sin_beta2}")
    print(f"cos(mu1): {cos_mu1}")
    print(f"cos(mu2): {cos_mu2}")
    print(f"alpha: {alpha}")
