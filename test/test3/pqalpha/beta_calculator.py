import numpy as np
from scipy.optimize import root_scalar

def calculate_theta(alpha, p, q, epsilon=1e-10):
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    def theta_equation(theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # 重新计算sin(2*theta)和cos(2*theta)
        sin_2theta = np.sin(2 * theta)
        cos_2theta = np.cos(2 * theta)

        # 计算cos_mu1和cos_mu2
        numerator1 = p00 + p10 * cos_theta
        denominator1 = np.sqrt(numerator1**2 + (p10 * sin_theta * sin_2theta)**2)
        if denominator1 < epsilon:
            return None
        cos_mu1 = numerator1 / denominator1

        numerator2 = p01 - p11 * cos_theta
        denominator2 = np.sqrt(numerator2**2 + (p11 * sin_theta * sin_2theta)**2)
        if denominator2 < epsilon:
            return None
        cos_mu2 = numerator2 / denominator2

        # 根据给定的alpha值计算需要的方程
        sin_mu1 = np.sqrt(1 - cos_mu1**2)
        sin_mu2 = - np.sqrt(1 - cos_mu2**2)  # 负号放在这里

        alpha_calc = (p10 * sin_mu1 - p11 * sin_mu2) * sin_theta * (cos_2theta / sin_2theta)

        return alpha_calc - alpha  # 返回与alpha差值的方程

    try:
        # 在区间(0, pi/4)内求解theta
        result = root_scalar(theta_equation, bracket=[0.001, np.pi/4 - 0.001], method='brentq')
        if result.converged:
            theta = result.root
            return theta
    except:
        pass

    return None
