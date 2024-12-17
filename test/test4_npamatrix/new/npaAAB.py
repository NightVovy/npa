import math

import cvxpy as cp

# Define variables
gamma = cp.Variable((11, 11),hermitian=True)  # Declare gamma as hermitian
# TODO: 13X13

# Define constraints
constraints = [
    gamma >> 0,  # Positive semidefinite constraint
    gamma == cp.conj(gamma.T),  # 共轭转置对称
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

    gamma[2, 5] == gamma[1, 7],  # A0A1B0
    gamma[2, 6] == gamma[1, 8],  # A0A1B1
    gamma[4, 5] == gamma[3, 6],  # A0B0B1
    gamma[3, 8] == gamma[4, 7],  # A1B0B1

    gamma[5, 7] == gamma[6, 8],  # X4

# New constraints for the extended 11x11 matrix
    gamma[0, 9] == gamma[3, 5],  # 1st row, 10th column == 3rd row, 6th column
    gamma[0, 10] == gamma[3, 6],  # 1st row, 11th column == 3rd row, 7th column
    gamma[3, 10] == gamma[4, 9],  # 4th row, 11th column == 5th row, 10th column
    gamma[5, 9] == gamma[6, 10],  # 6th row, 10th column == 7th row, 11th column
    gamma[7, 9] == gamma[8, 10],  # 8th row, 10th column == 9th row, 11th column
    gamma[6, 9] == gamma[5, 10],  # 7th row, 10th column == 6th row, 11th column
    gamma[8, 9] == gamma[7, 10],  # 9th row, 10th column == 8th row, 11th column
    gamma[7, 8] == gamma[9, 10],  # 8th row, 9th column == 10th row, 11th column
    gamma[3, 9] == gamma[4, 10],  # 4th row, 10th column == 5th row, 11th column

# New additional constraints
#     gamma[1, 9] == gamma[2, 3],  # 2nd row, 10th column == 3rd row, 4th column A1B0
#     gamma[1, 10] == gamma[2, 4],  # 2nd row, 11th column == 3rd row, 5th column A1B1
#     gamma[2, 9] == gamma[1, 3],  # 3rd row, 10th column == 2nd row, 4th column A0B0
#     gamma[2, 10] == gamma[3, 4],  # 3rd row, 11th column == 4th row, 5th column A0B1
]


# Objective function
# objective = cp.Maximize(
#     gamma[1, 3] + gamma[1, 4] +  gamma[2, 3] - gamma[2, 4]
# )
# 目标函数（确保它是实数值的）
objective = cp.Maximize(
    cp.real(gamma[1, 3]) + cp.real(gamma[1, 4]) + cp.real(gamma[2, 3]) - cp.real(gamma[2, 4])
)

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver="SDPA")  # Use "SDPA" or "MOSEK" as appropriate


print("Optimal matrix X:", gamma.value)
print("Optimal value:", problem.value)
# 输出gamma[1, 3], gamma[1, 4], gamma[2, 3], gamma[2, 4]
print(f"gamma[1, 3]: {gamma.value[1, 3]}")
print(f"gamma[1, 4]: {gamma.value[1, 4]}")
print(f"gamma[2, 3]: {gamma.value[2, 3]}")
print(f"gamma[2, 4]: {gamma.value[2, 4]}")

# Output 2 * sqrt(2)
sqrt_value = 2 * math.sqrt(2)
print(f"2 * sqrt(2): {sqrt_value}")