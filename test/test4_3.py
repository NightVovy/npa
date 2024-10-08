import numpy as np


def find_cosbeta():
    while True:
        p = np.random.uniform(0, 1)
        q = np.random.uniform(0, 1)

        p00 = p * q
        p01 = p * (1 - q)
        p10 = (1 - p) * q
        p11 = (1 - p) * (1 - q)

        def equation(cosbeta):
            return p10 * (p01 - p11 * cosbeta) ** 3 + p11 * (p00 + p10 * cosbeta) ** 3

        cosbeta_range = np.linspace(-1, 1, 1000)
        for cosbeta in cosbeta_range:
            if np.isclose(equation(cosbeta), 0, atol=1e-6):
                A = (p00 + p10 * cosbeta) ** 2
                B = (p01 - p11 * cosbeta) ** 2
                if B >= A:
                    return p, q, cosbeta, p00, p01, p10, p11, A, B


p, q, cosbeta, p00, p01, p10, p11, A, B = find_cosbeta()

print(f"p: {p}")
print(f"q: {q}")
print(f"cosbeta: {cosbeta}")
print(f"A: {A}")
print(f"B: {B}")
print(f"B - A: {B - A}")

C = (p10 ** 2 - p11 ** 2) * A * B / (p10 ** 2 * p11 ** 2 * (B - A))
lambda_value = np.sqrt(A + p10 ** 2 * C) + np.sqrt(B + p11 ** 2 * C)

print(f"lambda: {lambda_value}")
