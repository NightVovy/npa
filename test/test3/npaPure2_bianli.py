import cvxpy as cp
import numpy as np
import sys

# Define the ranges and step sizes for p, q, and alpha
p_values = np.arange(0.05, 1.05, 0.05)
q_values = np.arange(0.05, 1.05, 0.05)
alpha_values = np.arange(0.1, 2.1, 0.1)

# Store results in a list
results = []
valid_combinations = []  # To store the valid (p, q, alpha) combinations

# Open the file in write mode
with open('npa_bianli.txt', 'w') as f:
    # Save the original stdout (console output)
    original_stdout = sys.stdout
    # Redirect stdout to the file
    sys.stdout = f

    # Loop over all combinations of p, q, and alpha
    for p in p_values:
        for q in q_values:
            # Skip the case when q == 0.5
            if q == 0.5:
                continue

            for alpha in alpha_values:
                # Define parameters based on p and q
                p00 = p * q
                p01 = p * (1 - q)
                p10 = (1 - p) * q
                p11 = (1 - p) * (1 - q)

                # Define variables
                gamma = cp.Variable((9, 9))  # Declare gamma as Hermitian

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

                # Check if the output values meet the specified conditions
                if (
                    abs(gamma.value[1, 3]) > 0.98 and
                    abs(gamma.value[1, 4]) > 0.98 and
                    abs(gamma.value[2, 3]) > 0.98  # All three values are > 0.98
                ):
                    continue  # Skip this iteration if the condition is met

                # Check other conditions
                if (
                    abs(gamma.value[1, 3]) > 0.98 or  # |gamma[1, 3]| > 0.98
                    gamma.value[1, 3] < 0 or         # gamma[1, 3] < 0
                    gamma.value[1, 4] < 0 or         # gamma[1, 4] < 0
                    gamma.value[2, 3] < 0 or         # gamma[2, 3] < 0
                    gamma.value[2, 4] > 0            # gamma[2, 4] > 0
                ):
                    continue  # Skip this iteration if any of the condition is met

                # If the conditions are met, store the results
                result = {
                    'p': p,
                    'q': q,
                    'alpha': alpha,
                    'gamma_0_1': gamma.value[0, 1],
                    'A0B0': gamma.value[1, 3],
                    'A0B1': gamma.value[1, 4],
                    'A1B0': gamma.value[2, 3],
                    'A1B1': gamma.value[2, 4],
                    'gamma_matrix': gamma.value,
                    'objective_value': problem.value
                }

                # Append the result to the results list
                results.append(result)

                # Store the valid combinations (p, q, alpha) that meet the conditions
                valid_combinations.append((p, q, alpha))

    # Restore the original stdout (console output)
    sys.stdout = original_stdout

# Output all the valid (p, q, alpha) combinations that passed the conditions
print("Valid (p, q, alpha) combinations:")
for p, q, alpha in valid_combinations:
    print(f"p={p}, q={q}, alpha={alpha}")

# Also print the results to the console (optional)
for result in results:
    print(f"Current p={result['p']}, q={result['q']}, alpha={result['alpha']}:")
    print(f"A0={result['gamma_0_1']}")
    print(f"A0B0={result['A0B0']}")
    print(f"A0B1={result['A0B1']}")
    print(f"A1B0={result['A1B0']}")
    print(f"A1B1={result['A1B1']}")
    print("\nOptimal gamma matrix:")
    print(result['gamma_matrix'])
    print(f"Optimal value: {result['objective_value']}\n")
