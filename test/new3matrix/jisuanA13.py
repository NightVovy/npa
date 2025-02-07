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
p00 = 0.594745
p01 = 0.565932
p10 = 0.372355
p11 = 0.372355
beta2 = 0.222639
theta = 0.187306

# 计算第二个公式的值
result = calculate_second_formula(p00, p01, p10, p11, beta2, theta)

# 输出结果
print("The value of the second formula is:", result)
