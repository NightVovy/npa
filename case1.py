from ncpol2sdpa import *
import numpy as np


def generate_values(step=0.01):
    p00, p01, p10, p11 = 0, 0, 0, 0
    while True:
        p00 = np.random.uniform(step, 1)
        p01 = np.random.uniform(step, 1)
        p10 = np.random.uniform(step, 1)
        p11 = np.random.uniform(step, 1)

        if 1 / p00 + 1 / p01 + 1 / p10 - 1 / p11 > 0:
            break

    return p00, p01, p10, p11


# Example usage
p00, p01, p10, p11 = generate_values()
print(f"p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}")


level = 1

P = Probability([2, 2], [2, 2])

func = -P([0], [0], 'A') + P([0, 0], [0, 0]) + P([0, 0], [0, 1]) + \
       P([0, 0], [1, 0]) - P([0, 0], [1, 1]) - P([0], [0], 'B')

objective = -func