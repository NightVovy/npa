import numpy as np
from scipy.optimize import fsolve

# Known values
p00 = 0.3
p01 = 0.9
p10 = 2
p11 = 0.5
alpha = 1

# Define equations
def equations(vars, p00, p01, p10, p11, alpha):
    beta2, theta = vars
    A = p00 + p10 * np.cos(beta2)
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * np.cos(beta2)
    D = p11 * np.sin(beta2) * np.sin(2 * theta)

    if abs(np.cos(beta2)) >= 1 or abs(np.cos(2 * theta)) >= 1:
        return [np.inf, np.inf]  # Return infinity if conditions are not met

    eq1 = (p10 * np.sin(beta2) * A / np.sqrt(A ** 2 + B ** 2)) - (p11 * np.sin(beta2) * C / np.sqrt(C ** 2 + D ** 2))
    eq2 = ((p10 ** 2 * np.sin(beta2) / np.sqrt(A ** 2 + B ** 2)) + (p11 ** 2 * np.sin(beta2) / np.sqrt(C ** 2 + D ** 2))) * np.cos(2 * theta) - alpha

    return [eq1, eq2]

def update(p00, p01, p10, p11, alpha):
    initial_guess = [0.5, 0.5]
    beta2, theta = fsolve(equations, initial_guess, args=(p00, p01, p10, p11, alpha), maxfev=10000)
    cos_beta2 = np.cos(beta2)
    A = p00 + p10 * cos_beta2
    B = p10 * np.sin(beta2) * np.sin(2 * theta)
    C = p01 - p11 * cos_beta2
    D = p11 * np.sin(beta2) * np.sin(2 * theta)
    cos_2theta = np.cos(2 * theta)
    lambda_val = np.sqrt(A ** 2 + B ** 2) + np.sqrt(C ** 2 + D ** 2) + alpha * cos_2theta
    # print(f"beta2: {beta2}, theta: {theta}, cos(beta2): {cos_beta2}, A: {A}, B: {B}, C: {C}, D: {D}, cos(2*theta): {cos_2theta}")
    print(f"p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}, alpha: {alpha}")
    print(f"A: {A}, B: {B}, C: {C}, D: {D}")
    print(f"beta2: {beta2}, theta: {theta}")
    print(f"cos(2*theta): {cos_2theta}, cos(beta2): {cos_beta2}")
    print(f"lambda: {lambda_val}")

# Call the update function
update(p00, p01, p10, p11, alpha)
