import numpy as np
from scipy.optimize import fsolve

def equation(cosbeta, p00, p01, p10, p11):
    return p10 * (p01 - p11 * cosbeta)**3 + p11 * (p00 + p10 * cosbeta)**3

def solve_equation():
    solutions = []
    for _ in range(10):
        p = np.random.uniform(0, 0.5)  # 生成较小的p
        q = np.random.uniform(0.5, 1)  # 生成较大的q
        p00 = p * q
        p01 = p * (1 - q)
        p10 = (1 - p) * q
        p11 = (1 - p) * (1 - q)

        cosbeta_initial_guess = np.random.uniform(-1, 1)  # 随机初始猜测值
        cosbeta_solution = fsolve(equation, cosbeta_initial_guess, args=(p00, p01, p10, p11), maxfev=10000)
        cosbeta_solution = cosbeta_solution[0]  # 获取解的第一个元素
        if -1 <= cosbeta_solution <= 1:
            sinbeta_solution = np.sqrt(1 - cosbeta_solution**2)
            A = (p00 + p10 * cosbeta_solution)**2
            B = (p01 - p11 * cosbeta_solution)**2
            numerator = (p10**2 - p11**2) * A * B
            denominator = p10**2 * p11**2 * (B - A)
            solutions.append((p00, p01, p10, p11, cosbeta_solution, sinbeta_solution, A, B, numerator, denominator))
    return solutions

# 输出结果
results = solve_equation()

# 打印表头
print(f"{'p00':<10}{'p01':<10}{'p10':<10}{'p11':<10}{'cosbeta':<10}{'sinbeta':<10}{'A':<10}{'B':<10}{'numerator':<15}{'denominator':<15}")
print("-" * 110)

# 打印每一行结果
for result in results:
    p00, p01, p10, p11, cosbeta, sinbeta, A, B, numerator, denominator = result
    print(f"{p00:<10.4f}{p01:<10.4f}{p10:<10.4f}{p11:<10.4f}{cosbeta:<10.4f}{sinbeta:<10.4f}{A:<10.4f}{B:<10.4f}{numerator:<15.4f}{denominator:<15.4f}")
