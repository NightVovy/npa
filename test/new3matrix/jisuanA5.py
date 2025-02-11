import numpy as np


# 计算 sin(mu1), sin(mu2)
def compute_sin_mu(beta2, p00, p01, p10, p11, theta):
    sin_mu1 = (p10 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    sin_mu2 = - (p11 * np.sin(beta2) * np.sin(2 * theta)) / np.sqrt(
        (p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return sin_mu1, sin_mu2


# 计算 alpha
def calculate_alpha(p00, p01, p10, p11, beta2, theta):
    sin_mu1, sin_mu2 = compute_sin_mu(beta2, p00, p01, p10, p11, theta)

    alpha = (p10 * sin_mu1 - p11 * sin_mu2) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta))
    return alpha


# 用户输入
p00 = 0.243308500358061
p01 = 0.53948855550134
p10 = 0.8287013247435185
p11 = 0.52587900858036
beta2 = 1.41016048228704
theta = 0.71875697308974

# 计算 alpha
result = calculate_alpha(p00, p01, p10, p11, beta2, theta)

# 输出结果
print("The value of alpha is:", result)
