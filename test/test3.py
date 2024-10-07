import numpy as np
from sympy import symbols, Eq, solve, cos, sin, sqrt

# 定义已知变量
p00 = 0.9695846277645586
p01 = 0.9394989415641891
p10 = 0.8948273504276488
p11 = 0.7751328233611146
cos_beta1 = 1
sin_beta1 = 0
alpha = 1.1957999576221703

# 定义未知参数
theta, beta_2 = symbols('theta beta_2')

# 定义方程
# eq1 = Eq(alpha * cos_beta1 - \
#          (p10 * (p10 * sin(beta_2) * sin(2 * theta)) / sqrt(
#              (p00 + p10 * cos(beta_2)) ** 2 + (p10 * sin(beta_2) * sin(2 * theta)) ** 2) \
#           + p11 * (p11 * sin(beta_2) * sin(2 * theta) / sqrt(
#              (p01 - p11 * cos(beta_2)) ** 2 + (p11 * sin(beta_2) * sin(2 * theta)) ** 2)), 0))
eq1 = Eq(alpha * cos_beta1 - (p10 * (p10 * sin(beta_2) * sin(2 * theta)) / sqrt(
    (p00 + p10 * cos(beta_2)) ** 2 + (p10 * sin(beta_2) * sin(2 * theta)) ** 2)
                              + p11 * (p11 * sin(beta_2) * sin(2 * theta)) / sqrt(
            (p01 - p11 * cos(beta_2)) ** 2 + (p11 * sin(beta_2) * sin(2 * theta)) ** 2))
         * sin(beta_2) * cos(2 * theta) / sin(2 * theta), 0)

eq2 = Eq(p10 * sin(beta_2) * (p00 + p10 * cos(beta_2)) / sqrt(
    (p00 + p10 * cos(beta_2)) ** 2 + (p10 * sin(beta_2) * sin(2 * theta)) ** 2) -
         p11 * sin(beta_2) * (p01 - p11 * cos(beta_2)) / sqrt(
    (p01 - p11 * cos(beta_2)) ** 2 + (p11 * sin(beta_2) * sin(2 * theta)) ** 2), 0)

# 解方程
solution = solve((eq1, eq2), (theta, beta_2))

print(solution)
