import numpy as np
from scipy.optimize import fsolve

def solve(p, q, alpha):
    def equations(vars, p, q, alpha):
        beta2, theta = vars

        p00 = p * q
        p01 = p * (1 - q)
        p10 = (1 - p) * q
        p11 = (1 - p) * (1 - q)

        term1 = p10 * np.sin(beta2) * ((p00 + p10 * np.cos(beta2)) /
                                       np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

        term2 = p11 * np.sin(beta2) * ((p01 - p11 * np.cos(beta2)) /
                                       np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

        equation1 = term1 - term2

        term3 = p10 * ((p10 * np.sin(beta2) * np.sin(2 * theta)) /
                       np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

        term4 = p11 * ((-p11 * np.sin(beta2) * np.sin(2 * theta)) /
                       np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2))

        equation2 = (term3 - term4) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta)) - alpha

        return [equation1, equation2]

    def calculate_mu1_mu2(beta2, theta, p, q):
        p00 = p * q
        p01 = p * (1 - q)
        p10 = (1 - p) * q
        p11 = (1 - p) * (1 - q)

        cos_mu1 = (p00 + p10 * np.cos(beta2)) / np.sqrt(
            (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
        )
        sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
            (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2
        )

        cos_mu2 = (p01 - p11 * np.cos(beta2)) / np.sqrt(
            (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
        )
        sin_mu2 = -(p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
            (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2
        )

        return sin_mu1, cos_mu1, sin_mu2, cos_mu2

    # Solve for beta2 and theta
    initial_guess = [np.pi / 4, np.pi / 4]
    beta2, theta = fsolve(equations, initial_guess, args=(p, q, alpha))

    sin_beta2 = np.sin(beta2)
    cos_beta2 = np.cos(beta2)
    sin_2theta = np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)

    # Calculate mu1 and mu2 values
    sin_mu1, cos_mu1, sin_mu2, cos_mu2 = calculate_mu1_mu2(beta2, theta, p, q)
    

    # Print results
    # print(f"beta2: {beta2}, theta: {theta}, sin(beta2): {sin_beta2}, cos(beta2): {cos_beta2}, sin(2*theta): {sin_2theta}, cos(2*theta): {cos_2theta}, sin(mu1): {sin_mu1}, cos(mu1): {cos_mu1}, sin(mu2): {sin_mu2}, cos(mu2): {cos_mu2}")
    print(f"beta2: {beta2} radians, theta: {theta} radians")
    print(f"sin(beta2): {sin_beta2}, cos(beta2): {cos_beta2}, sin(2*theta): {sin_2theta}, cos(2*theta): {cos_2theta}")
    print(f"cos(mu1): {cos_mu1}, sin(mu1): {sin_mu1}")
    print(f"cos(mu2): {cos_mu2}, sin(mu2): {sin_mu2}")

    return beta2, theta, sin_beta2, cos_beta2, sin_2theta, cos_2theta, sin_mu1, cos_mu1, sin_mu2, cos_mu2

# 示例参数
# p = 0.5
# q = 0.4
# alpha = 0.8

# 调用 solve 函数
# results = solve(p, q, alpha)
