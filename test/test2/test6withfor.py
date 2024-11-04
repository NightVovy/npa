import numpy as np
import cvxpy as cp
from scipy.optimize import fsolve

# Define equations
def equations(vars, p00, p01, p10, p11, alpha):
    beta2, theta = vars
    A = p00 + p10 * np.cos(beta2)
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * np.cos(beta2)
    D = p11 * np.sin(beta2) * np.sin(2 * theta)

    # Ensure denominators are not zero and values inside sqrt are non-negative
    denom1 = np.sqrt(A ** 2 + B ** 2)
    denom2 = np.sqrt(C ** 2 + D ** 2)
    if denom1 == 0 or denom2 == 0 or A ** 2 + B ** 2 < 0 or C ** 2 + D ** 2 < 0:
        return [np.inf, np.inf]

    eq1 = (p10 * np.sin(beta2) * A / denom1) - (p11 * np.sin(beta2) * C / denom2)
    eq2 = ((p10 ** 2 * np.sin(beta2) / denom1) + (p11 ** 2 * np.sin(beta2) / denom2)) * np.cos(2 * theta) - alpha

    return [eq1, eq2]

def update(p00, p01, p10, p11, alpha):
    # Define variables
    gamma = cp.Variable((9, 9))

    # Define constraints
    constraints = [
        gamma >> 0,  # Positive semidefinite constraint
        gamma == cp.conj(gamma.T),  # Hermitian constraint
        cp.diag(gamma) == 1,  # Diagonal elements are 1

        gamma[0, 1] == gamma[3, 5],  # A0
        gamma[0, 1] == gamma[4, 6],

        gamma[0, 2] == gamma[3, 7],  # A1
        gamma[0, 2] == gamma[4, 8],

        gamma[0, 3] == gamma[1, 5],  # First row fourth element equals second row sixth element B0
        gamma[0, 3] == gamma[2, 7],

        gamma[0, 4] == gamma[2, 8],  # B1
        gamma[0, 4] == gamma[1, 6],

        gamma[1, 3] == gamma[0, 5],  # A0B0

        gamma[1, 4] == gamma[0, 6],  # A0B1

        gamma[2, 3] == gamma[0, 7],  # A1B0

        gamma[2, 4] == gamma[0, 8],  # A1B1

        gamma[1, 2] == gamma[5, 7],  # A0A1
        gamma[1, 2] == gamma[6, 8],
        gamma[3, 4] == gamma[5, 6],  # B0B1
        gamma[3, 4] == gamma[7, 8],

        gamma[2, 5] == gamma[1, 7],  # A0A1B0
        gamma[2, 6] == gamma[1, 8],  # A0A1B1
        gamma[4, 5] == gamma[3, 6],  # A0B0B1
        gamma[3, 8] == gamma[4, 7],  # A1B0B1

        gamma[5, 7] == gamma[6, 8],  # X4
    ]

    # Objective function
    objective = cp.Maximize(
        alpha * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4])

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="MOSEK")  # SDPA OR MOSEK

    initial_guess = [0.5, 0.5]
    beta2, theta = fsolve(equations, initial_guess, args=(p00, p01, p10, p11, alpha), maxfev=10000)
    cos_beta2 = np.cos(beta2)
    A = p00 + p10 * cos_beta2
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * cos_beta2
    D = p11 * np.sin(beta2) * np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    lambda_val = np.sqrt(A ** 2 + B ** 2) + np.sqrt(C ** 2 + D ** 2) + alpha * cos_2theta

    if abs(lambda_val - problem.value) < 1e-05:
        print(f"p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}, alpha: {alpha}")
        print(f"A: {A}, B: {B}, C: {C}, D: {D}")
        print(f"beta2: {beta2}, theta: {theta}")
        print(f"cos(2*theta): {cos_2theta}, cos(beta2): {cos_beta2}")
        print(f"lambda: {lambda_val}")
        print(f"Optimal value: {problem.value}")

# Iterate over all possible values
p_range = np.arange(0, 1.1, 0.1)
alpha_range = np.arange(0.1, 2.2, 0.2)

for p00 in p_range:
    for p01 in p_range:
        for p10 in p_range:
            for p11 in p_range:
                if p10 != p11:
                    for alpha in alpha_range:
                        print(f"Running update with p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}, alpha: {alpha}")
                        update(p00, p01, p10, p11, alpha)