import cvxpy as cp
import numpy as np

# Given p, q, alpha
p = 0.45  # Example value for p
q = 0.4  # Example value for q
alpha = 0.1  # Given alpha value

# Define parameters based on p and q
p00 = p * q
p01 = p * (1 - q)
p10 = (1 - p) * q
p11 = (1 - p) * (1 - q)

# Define variables
gamma = cp.Variable((9, 9), hermitian=True)  # Declare gamma as hermitian

# Define constraints
constraints = [
    gamma >> 0,  # Positive semidefinite constraint
    gamma == cp.conj(gamma.T),  # Hermitian symmetry
    cp.diag(gamma) == 1,  # Diagonal elements are 1

    gamma[0, 1] == gamma[3, 5],  # A0
    gamma[0, 1] == gamma[4, 6],

    gamma[0, 2] == gamma[3, 7],  # A1
    gamma[0, 2] == gamma[4, 8],

    gamma[0, 3] == gamma[1, 5],  # B0
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

    gamma[2, 5] == gamma[1, 7],  # A0A1B0 TODO: is this correct?
    gamma[2, 6] == gamma[1, 8],  # A0A1B1
    gamma[4, 5] == gamma[3, 6],  # A0B0B1
    gamma[3, 8] == gamma[4, 7],  # A1B0B1

    gamma[5, 7] == gamma[6, 8],  # X4
]

# Convert parameters to cvxpy Parameters for optimization
alpha_param = cp.Parameter(value=alpha)

# Objective function
# objective = cp.Maximize(
#     alpha_param * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4]
# )
# 目标函数（确保它是实数值的）
objective = cp.Maximize(
    alpha_param* cp.real(gamma[0, 1])+ p00 * cp.real(gamma[1, 3]) + p01 * cp.real(gamma[1, 4]) + p10 * cp.real(gamma[2, 3]) - p11 * cp.real(gamma[2, 4])
)

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver="SDPA")  # Use "SDPA" or "MOSEK" as appropriate

# Calculate the arcsin values for the relevant elements of the gamma matrix
arc_sin_sum = np.abs(
    np.arcsin(np.real(gamma.value[1, 3])) +
    np.arcsin(np.real(gamma.value[1, 4])) +
    np.arcsin(np.real(gamma.value[2, 3])) -
    np.arcsin(np.real(gamma.value[2, 4]))
)
# Print the result
print(f"Absolute value of arcsin expression: {arc_sin_sum}")

# Check the condition p00 - (p10 + p11) < p01 < p00 + (p10 + p11)
condition_left = p00 - (p10 + p11)
condition_right = p00 + (p10 + p11)

# Check if p01 satisfies the condition
condition_met = condition_left < p01 < condition_right

# Output the result of the condition check
print(f"Condition {condition_left} < p01 < {condition_right} is satisfied: {condition_met}")


# Print the entire gamma matrix
print("\nOptimal gamma matrix:")
print(gamma.value)

# Output the values of the gamma matrix in the required format
print(f"A0={gamma.value[0, 1]}")
print(f"A0B0={gamma.value[1, 3]}")
print(f"A0B1={gamma.value[1, 4]}")
print(f"A1B0={gamma.value[2, 3]}")
print(f"A1B1={gamma.value[2, 4]}")

# Print the optimal value of the objective
print(f"Optimal value: {problem.value}")
