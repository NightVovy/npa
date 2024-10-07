from scipy.optimize import fsolve
import numpy as np

# 定义已知变量
p00 = 0.9695846277645586
p01 = 0.9394989415641891
p10 = 0.8948273504276488
p11 = 0.7751328233611146
cos_beta1 = 1
sin_beta1 = 0
alpha = 1.1957999576221703


# 定义方程
def equations(vars):
    theta, beta_2 = vars
    # A5
    eq1 = alpha * cos_beta1 - (p10 * (p10 * np.sin(beta_2) * np.sin(2 * theta)) / np.sqrt(
        (p00 + p10 * np.cos(beta_2)) ** 2 + (p10 * np.sin(beta_2) * np.sin(2 * theta)) ** 2) +
                               p11 * (p11 * np.sin(beta_2) * np.sin(2 * theta)) / np.sqrt(
                (p01 - p11 * np.cos(beta_2)) ** 2 + (p11 * np.sin(beta_2) * np.sin(2 * theta)) ** 2)) * np.sin(beta_2) * np.cos(2*theta) / np.sin(2*theta)
    # A13
    eq2 = p10 * np.sin(beta_2) * (p00 + p10 * np.cos(beta_2)) / np.sqrt(
        (p00 + p10 * np.cos(beta_2)) ** 2 + (p10 * np.sin(beta_2) * np.sin(2 * theta)) ** 2) - \
          p11 * np.sin(beta_2) * (p01 - p11 * np.cos(beta_2)) / np.sqrt(
        (p01 - p11 * np.cos(beta_2)) ** 2 + (p11 * np.sin(beta_2) * np.sin(2 * theta)) ** 2)
    return [eq1, eq2]


# 初始猜测值
initial_guess = [1, 1]

# 使用fsolve求解方程
solution = fsolve(equations, initial_guess)

print(solution)
