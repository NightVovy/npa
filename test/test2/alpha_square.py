from scipy.optimize import minimize
import numpy as np

# 定义目标函数
def func(x):
    p00, p01, p10, p11 = x
    numerator = ((p11 ** 2) * ((p11 + p10) ** 2 - (p01 - p00) ** 2) ** 2 + 2 * p01 * p11 * (p11 + p10) * (p01 - p00))
    denominator = ((p11 ** 2) * (p01 - p00) ** 2 + (p01 ** 2) * (p11 + p10) ** 2)
    if denominator == 0:
        return np.inf  # 避免除以零，返回一个大数
    return numerator / denominator

# 定义约束条件
constraints = [{'type': 'ineq', 'fun': lambda x: 1 - abs((x[1] - x[0]) / (x[3] + x[2]))}]
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]

# 求最小值
res_min = minimize(func, x0=[0.5, 0.5, 0.5, 0.5], bounds=bounds, constraints=constraints)

# 求最大值，通过最大化负值实现
res_max = minimize(lambda x: -func(x), x0=[0.5, 0.5, 0.5, 0.5], bounds=bounds, constraints=constraints)

p00a = 1
p01a = 0
p10a = 1
p11a = 1

alpha2 = ((p11a ** 2) * ((p11a + p10a) ** 2 - (p01a - p00a) ** 2) ** 2 + 2 * p01a * p11a * (p11a + p10a) * (p01a - p00a)) / (
        (p11a ** 2) * (p01a - p00a) ** 2 + (p01a ** 2) * (p11a + p10a) ** 2)




print("最小值: ", res_min.fun)
print("最大值: ", -res_max.fun)
print("alpha2: ", alpha2)
