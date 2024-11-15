# 为了防止你猪头到又又又打错，这里写的是【绝对正确】的lambda1和lambda2
# 备注：还没检查是否正确，暂时不要用
import numpy as np
from scipy.optimize import fsolve


def solve_for_beta(alpha, theta, p00=1, p01=1, p10=1, p11=1):
    # Constants
    sin2theta = np.sin(2 * theta)

    # Define A and B as functions of beta (cos(beta))
    def equations(beta):
        cosbeta = np.cos(beta)
        A = (p00 + p10 * cosbeta) ** 2
        B = (p01 - p11 * cosbeta) ** 2
        sinbeta = np.sqrt(1 - cosbeta ** 2)

        # The equation to solve: (sin(beta))^2 / (sin(2*theta))^2
        lhs = (sinbeta ** 2) / (sin2theta ** 2)

        # Right hand side of the equation
        rhs = (p10 ** 2 - p11 ** 2) * A * B / (p10 ** 2 * p11 ** 2 * (B - A))

        return lhs - rhs  # The difference between left and right-hand sides

    # Use fsolve to find the root of the equation
    beta_guess = np.pi / 8  # Initial guess for beta
    beta_solution = fsolve(equations, beta_guess)[0]

    # Calculate sin(beta) and cos(beta)
    cosbeta = np.cos(beta_solution)
    sinbeta = np.sqrt(1 - cosbeta ** 2)

    # Output the results
    print(f"beta: {beta_solution}")
    print(f"sin(beta): {sinbeta}")
    print(f"cos(beta): {cosbeta}")

    return beta_solution, sinbeta, cosbeta


def calculate_lambda(alpha, theta, p00=1, p01=1, p10=1, p11=1):
    # Solve for beta and obtain sin(beta) and cos(beta)
    beta, sinbeta, cosbeta = solve_for_beta(alpha, theta, p00, p01, p10, p11)

    # F calculation
    F = ((p00 + p10 * cosbeta) / (p01 - p11 * cosbeta)) + 1

    # F2 calculation
    F2 = (p10 / p11) + 1

    # alpha2 calculation
    A = (p00 + p10 * cosbeta) ** 2
    B = (p01 - p11 * cosbeta) ** 2
    sin2theta = np.sin(2 * theta)
    cos2theta = np.cos(2 * theta)

    alpha2 = (p10 ** 2 * sinbeta * sin2theta / np.sqrt(A + p10 ** 2 * sinbeta ** 2 * sin2theta ** 2) +
              p11 ** 2 * sinbeta * sin2theta / np.sqrt(B + p11 ** 2 * sinbeta ** 2 * sin2theta ** 2)) * \
             sinbeta * cos2theta / sin2theta

    # lambda1 calculation
    lambda1 = F * np.sqrt((1 + alpha ** 2 / (p11 + p10) ** 2 - (p01 - p00) ** 2) *
                          (p11 ** 2 + p01 ** 2 - 2 * p11 * p01 * cosbeta))

    # Output the results
    print(f"F: {F}")
    print(f"F2: {F2}")
    print(f"alpha2: {alpha2}")
    print(f"lambda1: {lambda1}")

    return lambda1, F, F2, alpha2


# Example usage
alpha = 1.5  # Set alpha value in range [0, 2]
theta = np.pi / 8  # Set theta value in range (0, pi/4)
p00 = 0.5  # Set p00 value in range [0, 1]
p01 = 0.7  # Set p01 value in range [0, 1]
p10 = 1.0  # Set p10 value in range [0, 1]
p11 = 0.9  # Set p11 value in range [0, 1]

# Call the function to calculate lambda values and other parameters
calculate_lambda(alpha, theta, p00, p01, p10, p11)
