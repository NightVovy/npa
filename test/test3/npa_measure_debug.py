import numpy as np
from beta_calculator import calculate_beta2  # 导入calculate_beta2函数
from sdp_calculator import calculate_sdp  # 导入SDP计算函数
from measurement2 import quantum_measurement  # 导入量子测量函数

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx


# Function to calculate and print the required values based on p, q, theta
def calculate_results(p, q, theta):
    # Call the external beta2 calculation function
    beta2_data = calculate_beta2(p, q, theta)
    if beta2_data is None:
        print("No valid beta2 solution found.")
        return  # Skip if no valid beta2 solution is found

    beta2, cos_mu1, cos_mu2, alpha, sin_mu1, sin_mu2 = beta2_data

    # Call the external SDP calculation function
    gamma_matrix, problem_value = calculate_sdp(p, q, alpha)

    # Get measurement results from quantum_measurement function
    A0_measurement, A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement = quantum_measurement(
        beta2, cos_mu1, cos_mu2, theta)

    # Extract gamma matrix values
    A0 = gamma_matrix[0, 1]
    A0B0 = gamma_matrix[1, 3]
    A0B1 = gamma_matrix[1, 4]
    A1B0 = gamma_matrix[2, 3]
    A1B1 = gamma_matrix[2, 4]

    # Calculate p00, p01, p10, p11 based on given relationships
    p00 = p * q
    p01 = p * (1 - q)
    p10 = (1 - p) * q
    p11 = (1 - p) * (1 - q)

    # Calculate Iap
    Iap = alpha * A0_measurement + p00 * A0B0_measurement + p01 * A0B1_measurement + p10 * A1B0_measurement - p11 * A1B1_measurement

    # Output the results
    print(f"Results for p={p}, q={q}, theta={theta}:")
    print(f"p00={p00}, p01={p01}, p10={p10}, p11={p11}")
    print(f"beta2={beta2}, cos_mu1={cos_mu1}, cos_mu2={cos_mu2}, sin_mu1={sin_mu1}, sin_mu2={sin_mu2}, alpha={alpha}")
    print(f"Gamma matrix:\n{gamma_matrix}")
    print(f"Problem value: {problem_value}")        # TODO: NOT SAME?
    print(f"Iap: {Iap}")
    print(f"A0={A0},A0B0={A0B0}, A0B1={A0B1}, A1B0={A1B0}, A1B1={A1B1}")
    print(
        f"A0_measurement={A0_measurement}, A0B0_measurement={A0B0_measurement}, A0B1_measurement={A0B1_measurement}, A1B0_measurement={A1B0_measurement}, A1B1_measurement={A1B1_measurement}")

    # Check and print the differences
    print(f"Difference in A0B0: {A0B0 - A0B0_measurement}")
    print(f"Difference in A0B1: {A0B1 - A0B1_measurement}")
    print(f"Difference in A1B0: {A1B0 - A1B0_measurement}")
    print(f"Difference in A1B1: {A1B1 - A1B1_measurement}")


# Example usage: manually input p, q, and theta values
p = 0.4
q = 0.5
theta = 0.6
# theta max 0.7853981633974483

calculate_results(p, q, theta)
