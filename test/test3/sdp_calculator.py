import cvxpy as cp
import numpy as np

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

def calculate_sdp(p, q, alpha):
    """
    计算SDP并返回gamma矩阵和lambda值。
    """
    # Define parameters based on p and q
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # Define the gamma matrix variable (9x9 matrix)
    gamma = cp.Variable((9, 9))

    # Define the SDP constraints
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
        gamma[2, 5] == gamma[1, 7],  # A0A1B0
        gamma[2, 6] == gamma[1, 8],  # A0A1B1
        gamma[4, 5] == gamma[3, 6],  # A0B0B1
        gamma[3, 8] == gamma[4, 7],  # A1B0B1
        gamma[5, 7] == gamma[6, 8],  # X4
    ]

    # Convert parameters to cvxpy Parameters for optimization
    alpha_param = cp.Parameter(value=alpha)

    # Objective function
    objective = cp.Maximize(
        alpha_param * gamma[0, 1] + p00 * gamma[1, 3] + p01 * gamma[1, 4] + p10 * gamma[2, 3] - p11 * gamma[2, 4]
    )

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SDPA")  # Use "SDPA" or "MOSEK" as appropriate

    return gamma.value, problem.value  # Return the gamma matrix and optimal value

