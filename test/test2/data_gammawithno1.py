import cvxpy as cp
import numpy as np

# Define the function to solve the SDP problem for given parameters
def solve_sdp(alpha, p00, p01, p10, p11):
    # Define variables
    gamma = cp.Variable((9, 9))  # Declare gamma as hermitian

    # Define constraints
    constraints = [
        gamma >> 0,  # Positive semidefinite constraint
        gamma == cp.conj(gamma.T),  # Conjugate transpose symmetry
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
    p00_param = cp.Parameter(value=p00)
    p01_param = cp.Parameter(value=p01)
    p10_param = cp.Parameter(value=p10)
    p11_param = cp.Parameter(value=p11)

    # Objective function
    objective = cp.Maximize(
        alpha_param * gamma[0, 1] + p00_param * gamma[1, 3] + p01_param * gamma[1, 4] + p10_param * gamma[2, 3] - p11_param * gamma[2, 4]
    )

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SDPA")  # Use "SDPA" or "MOSEK" as appropriate

    return gamma.value, problem.value


# Read the data from file and process each line
with open('data.txt', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and split the line into key-value pairs
        line = line.strip()

        # Debug: Print the line being processed
        print("Processing line:", line)

        try:
            # Split line by commas and then by '=' to get key-value pairs
            params = dict((item.split('=')[0].strip(), item.split('=')[1].strip()) for item in line.split(','))

            # Debug: Print the parsed parameters
            print("Parsed parameters:", params)

            # Extract values
            p00 = float(params['p00'])
            p01 = float(params['p01'])
            p10 = float(params['p10'])
            p11 = float(params['p11'])
            alpha = float(params['alpha'])

            # Solve the SDP problem with the current parameters
            gamma_matrix, sdp_value = solve_sdp(alpha, p00, p01, p10, p11)

            # Debug: Print gamma_matrix
            print("Optimal gamma matrix:")
            print(gamma_matrix)

            # Iterate over all non-diagonal elements in the gamma matrix
            skip_line = False
            for i in range(gamma_matrix.shape[0]):
                for j in range(gamma_matrix.shape[1]):
                    if i != j:  # Skip diagonal elements
                        # Check if the absolute value of the non-diagonal element is too close to 1
                        if np.abs(gamma_matrix[i, j] - 1) < 1e-6:  # Check if difference with 1 is smaller than tolerance
                            print(f"Skipping line due to gamma[{i}, {j}] = {gamma_matrix[i, j]} too close to 1")
                            skip_line = True
                            break
                if skip_line:
                    break

            if skip_line:
                continue  # Skip the current line if any off-diagonal element is too close to 1

            # Output the result if the condition is not met
            print(f"Results for p00={p00}, p01={p01}, p10={p10}, p11={p11}, alpha={alpha}:")
            print("SDP value:", sdp_value)
            print("-" * 50)

        except KeyError as e:
            # Handle the error if a key is missing
            print(f"KeyError: Missing key {e} in line: {line}")
