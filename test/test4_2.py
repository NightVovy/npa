import sympy as sp
import numpy as np

# 定义变量
p, q, cosbeta = sp.symbols('p q cosbeta')

# 定义方程
p00 = p * q
p01 = p * (1 - q)
p10 = (1 - p) * q
p11 = (1 - p) * (1 - q)

equation = sp.Eq(p10 * (p01 - p11 * cosbeta) ** 3 + p11 * (p00 + p10 * cosbeta) ** 3, 0)

# 生成5次不同的p, q
np.random.seed(0)  # 固定随机种子以便复现结果
p_values = np.random.uniform(0.8, 1, 50)
q_values = np.random.uniform(0, 0.2, 50)

for p_val, q_val in zip(p_values, q_values):
    # 代入具体的p, q值
    eq = equation.subs({p: p_val, q: q_val})

    # 求解cosbeta
    cosbeta_solutions = sp.solve(eq, cosbeta)

    print(f"p = {p_val}, q = {q_val}")
    print(f"cosbeta_solutions = {cosbeta_solutions}")

    for cosbeta_sol in cosbeta_solutions:
        if cosbeta_sol.is_real and -1 <= cosbeta_sol <= 1:
            # 计算A和B
            A = (p00 + p10 * cosbeta_sol) ** 2
            B = (p01 - p11 * cosbeta_sol) ** 2

            # 计算C
            C = ((p10 ** 2 - p11 ** 2) * A * B) / (p10 ** 2 * p11 ** 2 * (B - A))

            # 计算lambda
            lambda_val = sp.sqrt(A + p10 ** 2 * C) + sp.sqrt(B + p11 ** 2 * C)

            # 计算beta
            beta = sp.acos(cosbeta_sol)

            # 计算具体数值
            A_val = A.subs({p: p_val, q: q_val, cosbeta: cosbeta_sol}).evalf()
            B_val = B.subs({p: p_val, q: q_val, cosbeta: cosbeta_sol}).evalf()
            C_val = C.subs({p: p_val, q: q_val, cosbeta: cosbeta_sol}).evalf()
            lambda_val = lambda_val.subs({p: p_val, q: q_val, cosbeta: cosbeta_sol}).evalf()
            beta_val = beta.evalf()

            # 输出结果
            print(f"p = {p_val}, q = {q_val}")
            print(
                f"p00 = {p_val * q_val}, p01 = {p_val * (1 - q_val)}, p10 = {(1 - p_val) * q_val}, p11 = {(1 - p_val) * (1 - q_val)}")
            print(f"cosbeta = {cosbeta_sol.evalf()}, beta = {beta_val}")
            print(f"A = {A_val}, B = {B_val}")
            print(f"C = {C_val}, lambda = {lambda_val}")
            print("\n")
