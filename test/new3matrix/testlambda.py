import numpy as np

# Define the given constants
# alpha = 1
p00 = 1
p01 = 0.5
p10 = 0.5
p11 = 0.3
beta1 = 0
beta2 = np.pi / 4
mu1 = np.pi / 5
mu2 = np.pi / 6
theta = np.pi / 7

# 计算 alpha
def compute_alpha(p10, p11, mu1, mu2, beta2, theta):
    alpha = (p10 * np.sin(mu1) - p11 * np.sin(mu2)) * np.sin(beta2) * (np.cos(2 * theta) / np.sin(2 * theta))
    return alpha

alpha = compute_alpha(p10, p11, mu1, mu2, beta2, theta)

# Compute the expression for lambda
lambda_value = (
    ((p00 * np.cos(beta1) + p10 * np.cos(beta2)) * np.cos(mu1) +
     (p01 * np.cos(beta1) - p11 * np.cos(beta2)) * np.cos(mu2)) +
    ((p00 * np.sin(beta1) + p10 * np.sin(beta2)) * np.sin(mu1) +
     (p01 * np.sin(beta1) - p11 * np.sin(beta2)) * np.sin(mu2)) * np.sin(2 * theta) +
    alpha * np.cos(2 * theta) * np.cos(beta1)
)

new_lambda_value = (
    alpha * np.cos(beta1) +
    p00 * np.cos(beta1) * np.cos(mu1) +
    p00 * np.sin(beta1) * np.sin(mu1) * np.sin(theta) / np.cos(theta) +
    p01 * np.cos(beta1) * np.cos(mu2) +
    p01 * np.sin(beta1) * np.sin(mu2) * np.sin(theta) / np.cos(theta) +
    p10 * np.cos(beta2) * np.cos(mu1) +
    p10 * np.sin(beta2) * np.sin(mu1) * np.sin(theta) / np.cos(theta) -
    p11 * np.cos(beta2) * np.cos(mu2) -
    p11 * np.sin(beta2) * np.sin(mu2) * np.sin(theta) / np.cos(theta)
)


print(f"The value of λ is: {lambda_value}")
print(f"The value of the new λ is: {new_lambda_value}")