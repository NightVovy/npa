import numpy as np
from beta_calculator import calculate_beta2  # 导入calculate_beta2函数
from sdp_calculator import calculate_sdp  # 导入SDP计算函数
from measurement2 import quantum_measurement  # 导入量子测量函数

# Pauli matrices
sigma_z = np.array([[1, 0], [0, -1]])  # σz
sigma_x = np.array([[0, 1], [1, 0]])  # σx

# Main loop to search for valid results
results = []
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
            A0_measurement, A0B0_measurement, A0B1_measurement, A1B0_measurement, A1B1_measurement = quantum_measurement(beta2, cos_mu1, cos_mu2, theta)

            # Extract gamma matrix values
            A0B0 = gamma_matrix[1, 3]
            A0B1 = gamma_matrix[1, 4]
            A1B0 = gamma_matrix[2, 3]
            A1B1 = gamma_matrix[2, 4]

            # Check for closeness of measurement values and gamma matrix values
            if np.isclose(A0B0, A0B0_measurement, atol=1e-3) and np.isclose(A0B1, A0B1_measurement, atol=1e-3) and \
               np.isclose(A1B0, A1B0_measurement, atol=1e-3) and np.isclose(A1B1, A1B1_measurement, atol=1e-3):
                results.append({
                    'p': p, 'q': q, 'theta': theta, 'beta2': beta2, 'alpha': alpha,
                    'cos_mu1': cos_mu1, 'cos_mu2': cos_mu2, 'A0B0': A0B0, 'A0B1': A0B1,
                    'A1B0': A1B0, 'A1B1': A1B1, 'gamma_matrix': gamma_matrix, 'problem_value': problem_value
                })

# Output all found results to a text file
with open('npa_measure2_5data.txt', 'w') as file:
    if results:
        file.write("Found matching results:\n")
        for result in results:
            file.write(f"p={result['p']}, q={result['q']}, theta={result['theta']}, beta2={result['beta2']}, alpha={result['alpha']}, "
                       f"cos_mu1={result['cos_mu1']}, cos_mu2={result['cos_mu2']}, A0B0={result['A0B0']}, A0B1={result['A0B1']}, "
                       f"A1B0={result['A1B0']}, A1B1={result['A1B1']}\n")
            file.write(f"Gamma matrix:\n{result['gamma_matrix']}\n")
            file.write(f"Optimal value: {result['problem_value']}\n\n")
    else:
        file.write("No results found.\n")