import numpy as np
from scipy.optimize import root_scalar

def calculate_beta2(p, q, theta, epsilon=1e-10):
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    def beta2_equation(beta2):
        sin_beta2 = np.sin(beta2)
        cos_beta2 = np.cos(beta2)

        numerator1 = p00 + p10 * cos_beta2
        denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator1 < epsilon:
            return None
        cos_mu1 = numerator1 / denominator1

        numerator2 = p01 - p11 * cos_beta2
        denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
        if denominator2 < epsilon:
            return None
        cos_mu2 = numerator2 / denominator2

        return p10 * sin_beta2 * cos_mu1 - p11 * sin_beta2 * cos_mu2

    try:
        # 设置求解区间为 (0, π)，避免包含 0 和 π
        result = root_scalar(beta2_equation, bracket=[0.1, np.pi - 0.1], method='brentq')
        if result.converged and 0 < result.root < np.pi - epsilon:  # 排除接近 0 和 π 的解
            beta2 = result.root
            cos_beta2 = np.cos(beta2)
            sin_beta2 = np.sin(beta2)

            numerator1 = p00 + p10 * cos_beta2
            denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu1 = numerator1 / max(denominator1, epsilon)

            numerator2 = p01 - p11 * cos_beta2
            denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu2 = numerator2 / max(denominator2, epsilon)

            # 计算 alpha
            sin_mu1 = np.sqrt(1 - cos_mu1**2)  # 计算 sin(mu1)
            sin_mu2 = np.sqrt(1 - cos_mu2**2)  # 计算 sin(mu2)

            alpha = (p10 * sin_mu1 - p11 * sin_mu2) * sin_beta2 * (np.cos(2 * theta) / np.sin(2 * theta))

            # 检查 alpha 是否在 (0, 2) 之间
            if 0 < alpha < 2:
                # 计算 mu1 和 mu2
                mu1 = np.arccos(cos_mu1)
                mu2 = np.arccos(cos_mu2)

                # 比较 mu1 和 mu2 与 pi/4
                is_mu1_greater_than_pi_over_4 = mu1 > np.pi / 4
                is_mu2_greater_than_pi_over_4 = mu2 > np.pi / 4

                return beta2, cos_mu1, cos_mu2, alpha, mu1, mu2, is_mu1_greater_than_pi_over_4, is_mu2_greater_than_pi_over_4
    except:
        pass

    return None

# 遍历 p, q, theta 的值
p_values = np.arange(0.1, 1.1, 0.1)
q_values = np.arange(0.1, 1.1, 0.1)
theta_values = np.arange(0.1, np.pi / 4, 0.05)

results = []
for p in p_values:
    for q in q_values:
        for theta in theta_values:
            result = calculate_beta2(p, q, theta)
            if result:
                beta2, cos_mu1, cos_mu2, alpha, mu1, mu2, is_mu1_greater_than_pi_over_4, is_mu2_greater_than_pi_over_4 = result
                results.append((p, q, theta, beta2, cos_mu1, cos_mu2, alpha, mu1, mu2, is_mu1_greater_than_pi_over_4, is_mu2_greater_than_pi_over_4))

# 输出符合条件的结果总数
print(f"\nTotal valid results: {len(results)}")
for res in results:
    print(f"p: {res[0]}, q: {res[1]}, theta: {res[2]}, beta2: {res[3]}, cos(mu1): {res[4]}, cos(mu2): {res[5]}, alpha: {res[6]}, mu1: {res[7]}, mu2: {res[8]}, is_mu1_greater_than_pi_over_4: {res[9]}, is_mu2_greater_than_pi_over_4: {res[10]}")
