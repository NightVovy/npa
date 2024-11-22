import numpy as np
from beta_calculator import calculate_beta2  # 导入calculate_beta2函数
from sdp_calculator import calculate_sdp  # 导入SDP计算函数
from measurement2 import quantum_measurement  # 导入量子测量函数

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

# Main loop to search for valid results
found_results = False
p_values = np.arange(0.1, 1.1, 0.1)
q_values = np.arange(0.1, 1.1, 0.1)
theta_values = np.arange(0.1, np.pi / 4, 0.05)

for p in p_values:
    for q in q_values:
        for theta in theta_values:
            # Call the external beta2 calculation function
            beta2_data = calculate_beta2(p, q, theta)
            if beta2_data is None:
                continue  # Skip if no valid beta2 solution is found

            beta2, cos_mu1, cos_mu2, alpha, sin_mu1, sin_mu2 = beta2_data

            # Call the external SDP calculation function
            gamma_matrix, problem_value = calculate_sdp(p, q, alpha)

            # Get measurement results from quantum_measurement function
            A0_measurement, A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement= quantum_measurement(beta2, cos_mu1, cos_mu2, theta)

            # Extract gamma matrix values
            A0B0 = gamma_matrix[1, 3]
            A0B1 = gamma_matrix[1, 4]
            A1B0 = gamma_matrix[2, 3]
            A1B1 = gamma_matrix[2, 4]

            # Check for closeness of measurement values and gamma matrix values
            if np.isclose(A0B0, A0B0_measurement, atol=1e-3) and np.isclose(A0B1, A0B1_measurement, atol=1e-3) and \
               np.isclose(A1B0, A1B0_measurement, atol=1e-3) and np.isclose(A1B1, A1B1_measurement, atol=1e-3):
                found_results = True
                print(f"Found matching results: p={p}, q={q}, theta={theta}, beta2={beta2}, alpha={alpha}, "
                      f"cos_mu1={cos_mu1}, cos_mu2={cos_mu2}, A0B0={A0B0}, A0B1={A0B1}, A1B0={A1B0}, A1B1={A1B1}")
                print(f"Gamma matrix:\n{gamma_matrix}")
                print(f"Optimal value: {problem_value}")
                break

            else:
                if not np.isclose(A0B0, A0B0_measurement, atol=1e-3):
                    print(f"Difference in A0B0: {A0B0 - A0B0_measurement}")
                if not np.isclose(A0B1, A0B1_measurement, atol=1e-3):
                    print(f"Difference in A0B1: {A0B1 - A0B1_measurement}")
                if not np.isclose(A1B0, A1B0_measurement, atol=1e-3):
                    print(f"Difference in A1B0: {A1B0 - A1B0_measurement}")
                if not np.isclose(A1B1, A1B1_measurement, atol=1e-3):
                    print(f"Difference in A1B1: {A1B1 - A1B1_measurement}")

        if found_results:
            break
    if found_results:
        break

# If no results were found
if not found_results:
    print("No results found.")
