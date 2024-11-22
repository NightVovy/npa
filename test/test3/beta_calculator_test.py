import numpy as np
from scipy.optimize import root_scalar

def calculate_beta2_v2(p, q, theta, epsilon=1e-10):
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
        result = root_scalar(beta2_equation, bracket=[0.1, np.pi - 0.1], method='brentq')
        if result.converged and 0 < result.root < np.pi - epsilon:
            beta2 = result.root
            cos_beta2 = np.cos(beta2)
            sin_beta2 = np.sin(beta2)

            # Original calculation for sin_mu1, sin_mu2
            numerator1 = p00 + p10 * cos_beta2
            denominator1 = np.sqrt(numerator1**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu1 = numerator1 / max(denominator1, epsilon)
            sin_mu1 = np.sqrt(1 - cos_mu1**2)

            numerator2 = p01 - p11 * cos_beta2
            denominator2 = np.sqrt(numerator2**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)
            cos_mu2 = numerator2 / max(denominator2, epsilon)
            sin_mu2 = - np.sqrt(1 - cos_mu2**2)

            # New calculation for sin_mu1, sin_mu2 using the given formula
            sin_mu1_v2 = (p10 * sin_beta2 * np.sin(2 * theta)) / np.sqrt((p00 + p10 * cos_beta2)**2 + (p10 * sin_beta2 * np.sin(2 * theta))**2)
            sin_mu2_v2 = -(p11 * sin_beta2 * np.sin(2 * theta)) / np.sqrt((p01 - p11 * cos_beta2)**2 + (p11 * sin_beta2 * np.sin(2 * theta))**2)

            # Output the comparison
            print(f"Original sin_mu1: {sin_mu1}, New sin_mu1: {sin_mu1_v2}")
            print(f"Original sin_mu2: {sin_mu2}, New sin_mu2: {sin_mu2_v2}")

            return beta2, cos_mu1, cos_mu2, sin_mu1, sin_mu2, sin_mu1_v2, sin_mu2_v2
    except:
        pass

    return None

# Test with given values p = 0.5, q = 0.6, theta = 0.7
p = 0.5
q = 0.6
theta = 0.7

result = calculate_beta2_v2(p, q, theta)
if result:
    beta2, cos_mu1, cos_mu2, sin_mu1, sin_mu2, sin_mu1_v2, sin_mu2_v2 = result
    print(f"Beta2: {beta2}, cos_mu1: {cos_mu1}, cos_mu2: {cos_mu2}")
else:
    print("No solution found.")
