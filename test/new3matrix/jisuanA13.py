import numpy as np

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

# 计算第二个公式左侧
def calculate_second_formula(p00, p01, p10, p11, beta2, theta):
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(beta2, p00, p01, p10, p11, theta)

    term1 = p10 * np.sin(beta2) * (p00 + p10 * np.cos(beta2)) / np.sqrt((p00 + p10 * np.cos(beta2)) ** 2 + (p10 * np.sin(beta2) * np.sin(2 * theta)) ** 2)
    term2 = -p11 * np.sin(beta2) * (p01 - p11 * np.cos(beta2)) / np.sqrt((p01 - p11 * np.cos(beta2)) ** 2 + (p11 * np.sin(beta2) * np.sin(2 * theta)) ** 2)

    return term1 + term2

# 用户输入
p00 = 0.243308500358061
p01 = 0.53948855550134
p10 = 0.8287013247435185
p11 = 0.52587900858036
beta2 = 1.41016048228704
theta = 0.71875697308974
# Result: 1.3809147373711372

# 计算第二个公式的值
result = calculate_second_formula(p00, p01, p10, p11, beta2, theta)

# 输出结果
print("The value of the second formula is:", result)
