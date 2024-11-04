import numpy as np

# Known values
p00 = 0.8
p01 = 0.6
p10 = 1
p11 = 1
beta2 = 1.514368218580490
theta = 2.5711416246329906

# Calculate A, B, C, D
A = p00 + p10 * np.cos(beta2)
B = p10 * np.sin(beta2) * np.sin(2 * theta)
C = p01 - p11 * np.cos(beta2)
D = p11 * np.sin(beta2) * np.sin(2 * theta)

# Calculate alpha
alpha = ((p10**2 * np.sin(beta2) / np.sqrt(A**2 + B**2)) +
          (p11**2 * np.sin(beta2) / np.sqrt(C**2 + D**2))) * np.cos(2 * theta)

# Print the result
print(f"Calculated alpha: {alpha}")
